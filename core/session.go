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
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// InferenceSession represents a running inference endpoint
type InferenceSession struct {
	ID             string
	ModelID        string
	CheckpointPath string // Added: Path to the actual model checkpoint
	ModelSize      string // Added: Model size (3B, 8B, 70B, etc.)
	Status         SessionStatus
	Endpoint       *url.URL
	Expiration     time.Time
	LastError      error
	MaxIdleTime    time.Duration
	IsPersistent   bool
	httpClient     *http.Client
}

// InferenceEndpointRequest represents the request body sent to inference endpoints
type InferenceEndpointRequest struct {
	Stream    bool      `json:"stream"`
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	MaxTokens int       `json:"max_tokens"`
}

// InferenceEndpointResponse represents a streaming response from the endpoint
type InferenceEndpointResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           int64  `json:"created"`
	Model             string `json:"model,omitempty"`
	SystemFingerprint string `json:"system_fingerprint,omitempty"`
	Choices           []struct {
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
			Timeout: 10 * time.Minute,
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

	// Create the request body for mindlet API
	reqBody := struct {
		ModelID        string    `json:"model_id"`
		CheckpointPath string    `json:"checkpoint_path,omitempty"`
		ModelSize      string    `json:"model_size,omitempty"`
		Messages       []Message `json:"messages"`
		MaxTokens      int       `json:"max_tokens"`
	}{
		ModelID:        s.ModelID,
		CheckpointPath: s.CheckpointPath,
		ModelSize:      s.ModelSize,
		Messages:       messages,
		MaxTokens:      1500,
	}

	// Serialize request body
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// Log request size for debugging
	log.Printf("Sending inference request: %d bytes (%.2f MB) with %d messages",
		len(jsonBody), float64(len(jsonBody))/(1024*1024), len(messages))

	// Create HTTP request to mindlet's streaming endpoint
	endpointURL := s.Endpoint.String()
	if !strings.HasSuffix(endpointURL, "/api/inference/stream") {
		if !strings.HasSuffix(endpointURL, "/") {
			endpointURL += "/"
		}
		endpointURL += "api/inference/stream"
	}

	log.Printf("Sending inference request to mindlet at %s for model %s", endpointURL, s.ModelID)
	if s.CheckpointPath != "" {
		log.Printf("Using checkpoint path: %s", s.CheckpointPath)
	}

	// Create a dedicated HTTP client for this request
	httpClient := &http.Client{
		Timeout: 30 * time.Minute, // Increase timeout for large requests
		Transport: &http.Transport{
			MaxIdleConns:           10,
			MaxIdleConnsPerHost:    10,
			IdleConnTimeout:        90 * time.Second,
			ResponseHeaderTimeout:  60 * time.Second,
			ExpectContinueTimeout:  1 * time.Second,
			MaxResponseHeaderBytes: 1 << 20, // 1 MB
			WriteBufferSize:        1 << 20, // 1 MB
			ReadBufferSize:         1 << 20, // 1 MB
			DisableCompression:     true,
		},
	}

	// Use a background context for the HTTP request to avoid cancellation issues
	req, err := http.NewRequestWithContext(context.Background(), "POST", endpointURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.ContentLength = int64(len(jsonBody))

	// Make the request
	startTime := time.Now()
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request after %v: %w", time.Since(startTime), err)
	}

	// Check response status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("mindlet endpoint returned error: %s - %s", resp.Status, string(body))
	}

	log.Printf("Got response from mindlet after %v, starting stream processing", time.Since(startTime))

	// Create a stream to handle the response
	stream, streamCtx := newBaseGenerationStream(ctx)

	// Start a goroutine to process the stream
	go func() {
		defer resp.Body.Close()
		defer stream.Stop()

		reader := bufio.NewReader(resp.Body)
		messageCount := 0
		totalBytes := 0

		for {
			select {
			case <-streamCtx.Done():
				log.Printf("Stream context cancelled after %d messages", messageCount)
				return
			default:
				// Set a read deadline to prevent hanging
				if conn, ok := resp.Body.(interface{ SetReadDeadline(time.Time) error }); ok {
					conn.SetReadDeadline(time.Now().Add(5 * time.Minute))
				}

				// Read SSE message
				sseMessage, err := s.readSSEMessage(reader)
				if err != nil {
					if err == io.EOF {
						log.Printf("Stream ended (EOF) after %d messages, %d bytes", messageCount, totalBytes)
						return
					}

					// Check if it's a timeout
					if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
						log.Printf("Read timeout after %d messages, closing stream", messageCount)
						return
					}

					log.Printf("Error reading SSE message after %d messages: %v", messageCount, err)
					if !strings.Contains(err.Error(), "context canceled") {
						stream.SendResponse(ModelResponse{Error: err})
					}
					return
				}

				if sseMessage == "" {
					continue
				}

				totalBytes += len(sseMessage)

				// Parse data lines from the SSE message
				lines := strings.Split(sseMessage, "\n")
				for _, line := range lines {
					line = strings.TrimSpace(line)
					if !strings.HasPrefix(line, "data:") {
						continue
					}

					dataContent := strings.TrimSpace(strings.TrimPrefix(line, "data:"))

					// Check for end of stream
					if dataContent == "[DONE]" {
						log.Printf("Received [DONE] signal after %d messages", messageCount)
						return
					}

					// Parse JSON response
					var streamResp InferenceEndpointResponse
					if err := json.Unmarshal([]byte(dataContent), &streamResp); err != nil {
						log.Printf("Error parsing JSON: %v, data: %s", err, dataContent)
						continue
					}

					// Process the response
					if len(streamResp.Choices) > 0 {
						choice := streamResp.Choices[0]

						if choice.FinishReason == "stop" {
							log.Printf("Received finish_reason: stop after %d messages", messageCount)
							return
						}

						if choice.Delta.Content != "" {
							messageCount++

							// Try to send with timeout
							sendStart := time.Now()
							if !stream.SendResponse(ModelResponse{Content: choice.Delta.Content}) {
								log.Printf("Failed to send response %d after %v, channel might be blocked or closed",
									messageCount, time.Since(sendStart))
								return
							}

							// Log progress every 100 messages
							if messageCount%100 == 0 {
								log.Printf("Processed %d messages, %d bytes total", messageCount, totalBytes)
							}
						}
					}
				}
			}
		}
	}()

	return stream, nil
}

// readSSEMessage reads a complete SSE message from the reader
func (s *InferenceSession) readSSEMessage(reader *bufio.Reader) (string, error) {
	var message strings.Builder
	hasData := false

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return "", err
		}

		// Check if this line contains data
		if strings.HasPrefix(line, "data:") {
			hasData = true
		}

		message.WriteString(line)

		// SSE messages are separated by empty lines
		// An empty line after data indicates end of message
		if line == "\n" && hasData {
			return strings.TrimSpace(message.String()), nil
		}

		// Also handle case where we have two consecutive newlines
		if strings.HasSuffix(message.String(), "\n\n") && hasData {
			return strings.TrimSpace(message.String()), nil
		}
	}
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

// SetCheckpointPath sets the checkpoint path for the session
func (s *InferenceSession) SetCheckpointPath(path string) {
	s.CheckpointPath = path
}

// SetModelSize sets the model size for the session
func (s *InferenceSession) SetModelSize(size string) {
	s.ModelSize = size
}
