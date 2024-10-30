package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"github.com/sashabaranov/go-openai"
	"net/http"
	"os"
)

// API Model IDs are defined by their last byte
const (
	APIModelIDGPT4          byte = 1
	APIModelIDGPT35Turbo    byte = 2
	APIModelIDClaude3Opus   byte = 3
	APIModelIDClaude3Sonnet byte = 4
	APIModelIDClaude3Haiku  byte = 5
)

// APIModelToDigest converts an API model ID to a full Digest
func APIModelToDigest(id byte) db.Digest {
	var d [32]byte
	d[31] = id // Set only the last byte
	return db.NewDigest(d[:])
}

// APIModelManager handles API-based external models
type APIModelManager struct {
	models *utils.ConcurrentMap[string, APIModel]
}

// APIModel represents external API-based models like GPT/Claude
type APIModel interface {
	GenerateStream(ctx context.Context, messages []Message) (GenerationStream, error)
}

type ModelResponse struct {
	Content string
	Error   error
}

func NewAPIModelManager() (*APIModelManager, error) {
	manager := &APIModelManager{
		models: utils.NewConcurrentMap[string, APIModel](),
	}

	if err := manager.initGPTModel(); err != nil {
		return nil, fmt.Errorf("failed to initialize GPT model: %w", err)
	}

	// TODO: Uncomment this once Claude API is available through a python service
	//if err := manager.initClaudeModel(); err != nil {
	//	return nil, fmt.Errorf("failed to initialize Claude model: %w", err)
	//}

	return manager, nil
}

func (m *APIModelManager) IsAPIModel(modelID db.Digest) bool {
	switch modelID.Bytes()[31] {
	case APIModelIDGPT4, APIModelIDGPT35Turbo,
		APIModelIDClaude3Opus, APIModelIDClaude3Sonnet, APIModelIDClaude3Haiku:
		return true
	default:
		return false
	}
}

func (m *APIModelManager) GetAPIModel(modelID db.Digest) (APIModel, error) {
	model, ok := m.models.Get(modelID.String())
	if !ok {
		return nil, fmt.Errorf("unknown API model ID: %d", modelID.Bytes()[31])
	}
	return model, nil
}

// Rest of the code remains the same, just update the environment variable names
func (m *APIModelManager) initGPTModel() error {
	apiKey := os.Getenv("GPT_API_KEY") // Changed from OPENAI_API_KEY to match your .env
	if apiKey == "" {
		return fmt.Errorf("GPT_API_KEY not set")
	}

	client := NewGPTClient(apiKey)
	m.models.Set(APIModelToDigest(APIModelIDGPT4).String(), client)
	m.models.Set(APIModelToDigest(APIModelIDGPT35Turbo).String(), client)
	return nil
}

func (m *APIModelManager) initClaudeModel() error {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("ANTHROPIC_API_KEY not set")
	}

	client := NewClaudeClient(apiKey)
	m.models.Set(APIModelToDigest(APIModelIDClaude3Opus).String(), client)
	m.models.Set(APIModelToDigest(APIModelIDClaude3Sonnet).String(), client)
	m.models.Set(APIModelToDigest(APIModelIDClaude3Haiku).String(), client)
	return nil
}

// GPTClient implements APIModel for OpenAI models
type GPTClient struct {
	client *openai.Client
}

func NewGPTClient(apiKey string) *GPTClient {
	return &GPTClient{
		client: openai.NewClient(apiKey),
	}
}

func (c *GPTClient) GenerateStream(ctx context.Context, messages []Message) (GenerationStream, error) {
	stream, streamCtx := newBaseGenerationStream(ctx)

	openaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		openaiMessages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	req := openai.ChatCompletionRequest{
		Model:    "gpt-4o-mini",
		Messages: openaiMessages,
		Stream:   true,
	}

	completion, err := c.client.CreateChatCompletionStream(streamCtx, req)
	if err != nil {
		stream.Stop()
		return nil, fmt.Errorf("failed to create chat completion stream: %w", err)
	}

	go func() {
		defer stream.Stop()
		defer completion.Close()

		for {
			select {
			case <-streamCtx.Done():
				return
			default:
				response, err := completion.Recv()
				if err != nil {
					stream.SendResponse(ModelResponse{Error: err})
					return
				}

				if len(response.Choices) > 0 {
					if response.Choices[0].Delta.Content != "" {
						if !stream.SendResponse(ModelResponse{
							Content: response.Choices[0].Delta.Content,
						}) {
							return
						}
					}

					// Handle completion
					if response.Choices[0].FinishReason == "stop" {
						return
					}
				}
			}
		}
	}()

	return stream, nil
}

// ClaudeClient implements APIModel for Anthropic models
type ClaudeClient struct {
	apiKey     string
	httpClient *http.Client
}

func NewClaudeClient(apiKey string) *ClaudeClient {
	return &ClaudeClient{
		apiKey:     apiKey,
		httpClient: &http.Client{},
	}
}

type ClaudeStreamResponse struct {
	Type  string `json:"type"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

func (c *ClaudeClient) GenerateStream(ctx context.Context, messages []Message) (GenerationStream, error) {
	stream, streamCtx := newBaseGenerationStream(ctx)

	// Convert messages to Claude format
	var systemMessage string
	var userMessages []string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemMessage = msg.Content
		case "user":
			userMessages = append(userMessages, msg.Content)
		}
	}

	// Combine messages
	prompt := systemMessage
	if len(userMessages) > 0 {
		if prompt != "" {
			prompt += "\n\n"
		}
		prompt += userMessages[len(userMessages)-1]
	}

	reqBody := map[string]interface{}{
		"model":    "claude-3-opus-20240229",
		"messages": []map[string]string{{"role": "user", "content": prompt}},
		"stream":   true,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		stream.Stop()
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(streamCtx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(jsonBody))
	if err != nil {
		stream.Stop()
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		stream.Stop()
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}

	go func() {
		defer resp.Body.Close()
		defer stream.Stop()

		decoder := json.NewDecoder(resp.Body)
		for {
			select {
			case <-streamCtx.Done():
				return
			default:
				var streamResp ClaudeStreamResponse
				if err := decoder.Decode(&streamResp); err != nil {
					stream.SendResponse(ModelResponse{Error: err})
					return
				}

				if streamResp.Type == "content_block_delta" && streamResp.Delta.Type == "text" {
					if !stream.SendResponse(ModelResponse{Content: streamResp.Delta.Text}) {
						return
					}
				}
			}
		}
	}()

	return stream, nil
}
