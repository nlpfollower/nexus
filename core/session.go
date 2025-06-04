// core/session.go
package core

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// InferenceSession represents a running inference endpoint
type InferenceSession struct {
	ID           string
	ModelID      string
	Status       SessionStatus
	Endpoint     *url.URL
	Expiration   time.Time
	LastError    error
	MaxIdleTime  time.Duration
	IsPersistent bool
	httpClient   *http.Client
}

// InferenceEndpointRequest represents the request body sent to inference endpoints
type InferenceEndpointRequest struct {
	Stream   bool      `json:"stream"`
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

// InferenceEndpointResponse represents a streaming response from the endpoint
type InferenceEndpointResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

// NewInferenceSession creates a new inference session
func NewInferenceSession(id string, modelID string, maxIdleTime time.Duration, isPersistent bool) *InferenceSession {
	return &InferenceSession{
		ID:           id,
		ModelID:      modelID,
		Status:       SessionStatusStopped,
		MaxIdleTime:  maxIdleTime,
		Expiration:   time.Now().Add(maxIdleTime),
		IsPersistent: isPersistent,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// ProcessInference sends an inference request to the session endpoint and returns a stream of responses
func (s *InferenceSession) ProcessInference(ctx context.Context, messages []Message) (GenerationStream, error) {
	if s.Status != SessionStatusRunning {
		return nil, fmt.Errorf("session is not running (status: %s)", s.Status)
	}

	if s.Endpoint == nil {
		return nil, fmt.Errorf("session has no endpoint URL")
	}

	// Create the request body
	reqBody := InferenceEndpointRequest{
		Stream:   true,
		Model:    s.ModelID,
		Messages: messages,
	}

	// Serialize request body
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// Create HTTP request
	endpointURL := s.Endpoint.String()
	if !strings.HasSuffix(endpointURL, "/v1/chat/completions") {
		if !strings.HasSuffix(endpointURL, "/") {
			endpointURL += "/"
		}
		endpointURL += "v1/chat/completions"
	}

	log.Printf("Sending inference request to %s", endpointURL)

	// Use a background context for the HTTP request to avoid cancellation issues
	req, err := http.NewRequestWithContext(context.Background(), "POST", endpointURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream") // For SSE responses

	// Make the request
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}

	// Check response status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("inference endpoint returned error: %s - %s", resp.Status, string(body))
	}

	// Create a stream to handle the response
	stream, streamCtx := newBaseGenerationStream(ctx)

	// Start a goroutine to process the stream
	go func() {
		defer resp.Body.Close()
		defer stream.Stop()

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // 1MB max line size

		for scanner.Scan() {
			select {
			case <-streamCtx.Done():
				return
			default:
				line := scanner.Text()

				// Skip empty lines
				if line == "" {
					continue
				}

				// Handle Server-Sent Events (SSE) format
				if strings.HasPrefix(line, "data: ") {
					// Remove the "data: " prefix
					jsonData := strings.TrimPrefix(line, "data: ")
					jsonData = strings.TrimSpace(jsonData)

					// Check for end of stream
					if jsonData == "[DONE]" {
						log.Printf("Received end of stream signal")
						return
					}

					// Parse the JSON response
					var streamResp InferenceEndpointResponse
					if err := json.Unmarshal([]byte(jsonData), &streamResp); err != nil {
						log.Printf("Error parsing SSE data: %v, data: %s", err, jsonData)
						continue
					}

					// Process the response
					if len(streamResp.Choices) > 0 {
						choice := streamResp.Choices[0]

						// Check for finish reason first
						if choice.FinishReason == "stop" {
							log.Printf("Received finish_reason: stop")
							return
						}

						// Extract content
						content := choice.Delta.Content
						if content != "" {
							// Send the content
							if !stream.SendResponse(ModelResponse{Content: content}) {
								return
							}
						}
					}
				}
			}
		}

		if err := scanner.Err(); err != nil {
			log.Printf("Scanner error: %v", err)
			// Don't send error responses for EOF or canceled context
			if err != io.EOF && !strings.Contains(err.Error(), "context canceled") {
				stream.SendResponse(ModelResponse{Error: err})
			}
		}
	}()

	return stream, nil
}

// Extend prolongs the session lifetime
func (s *InferenceSession) Extend(duration time.Duration) {
	s.Expiration = time.Now().Add(duration)
}

// TimeRemaining returns the time until the session expires
func (s *InferenceSession) TimeRemaining() time.Duration {
	remaining := time.Until(s.Expiration)
	if remaining < 0 {
		return 0
	}
	return remaining
}

// IsExpired checks if the session has expired
func (s *InferenceSession) IsExpired() bool {
	return time.Now().After(s.Expiration)
}
