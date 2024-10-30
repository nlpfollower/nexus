package core

import (
	"context"
	"github.com/joho/godotenv"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func init() {
	// Try to load .env from the project root first
	if err := godotenv.Load(filepath.Join("..", ".env")); err != nil {
		// If that fails, try current directory
		_ = godotenv.Load(".env")
	}
}

func TestLiveAPIModelManager(t *testing.T) {
	if os.Getenv("CI") == "true" {
		t.Skip("Skipping integration test in CI environment")
	}

	apiKey := os.Getenv("GPT_API_KEY")
	require.NotEmpty(t, apiKey, "GPT_API_KEY not set")

	// Set up a client we'll reuse across tests
	client := NewGPTClient(apiKey)

	t.Run("GPT-4 Integration", func(t *testing.T) {
		ctx := context.Background()
		messages := []Message{
			{Role: "system", Content: "You are a helpful assistant. Be concise."},
			{Role: "user", Content: "What is 2+2?"},
		}

		stream, err := client.GenerateStream(ctx, messages)
		require.NoError(t, err)

		var response string
		for msg := range stream.ResponseChan() {
			require.NoError(t, msg.Error)
			response += msg.Content
		}

		require.NotEmpty(t, response)
		require.Contains(t, response, "4")
		t.Logf("GPT-4 Response: %s", response)
	})

	t.Run("Multi-turn Conversation", func(t *testing.T) {
		ctx := context.Background()
		messages := []Message{
			{Role: "system", Content: "You are a helpful assistant. Always respond with exactly one word."},
			{Role: "user", Content: "What color is the sky?"},
			{Role: "assistant", Content: "Blue"},
			{Role: "user", Content: "What color is grass?"},
		}

		stream, err := client.GenerateStream(ctx, messages)
		require.NoError(t, err)

		var response string
		for msg := range stream.ResponseChan() {
			require.NoError(t, msg.Error)
			response += msg.Content
		}

		require.NotEmpty(t, response)
		words := len(response)
		require.Less(t, words, 10, "Response should be approximately one word")
		t.Logf("Multi-turn Response: %s", response)
	})

	t.Run("Error Cases", func(t *testing.T) {
		t.Run("Invalid API Key", func(t *testing.T) {
			badClient := NewGPTClient("invalid-key")
			ctx := context.Background()

			messages := []Message{
				{Role: "user", Content: "Hello"},
			}

			stream, err := badClient.GenerateStream(ctx, messages)
			if err != nil {
				// Some APIs might error immediately
				require.Contains(t, err.Error(), "auth")
				return
			}

			// If not errored immediately, should error in stream
			msg := <-stream.ResponseChan()
			require.Error(t, msg.Error)
			require.Contains(t, msg.Error.Error(), "auth")
		})

		t.Run("Context Cancellation", func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()

			messages := []Message{
				{Role: "user", Content: "Write a short essay about the history of humanity"},
			}

			stream, err := client.GenerateStream(ctx, messages)
			if err != nil {
				require.Contains(t, err.Error(), "context")
				return
			}

			msg := <-stream.ResponseChan()
			require.Error(t, msg.Error)
			require.Contains(t, msg.Error.Error(), "context")
		})

		t.Run("Stop Generation", func(t *testing.T) {
			ctx := context.Background()
			messages := []Message{
				{Role: "user", Content: "Write a short essay about the history of humanity"},
			}

			stream, err := client.GenerateStream(ctx, messages)
			require.NoError(t, err)

			// Read first message to ensure generation started
			msg := <-stream.ResponseChan()
			require.NoError(t, msg.Error)
			require.NotEmpty(t, msg.Content)

			// Stop the generation
			stream.Stop()

			// Verify the stream is properly closed by checking if ResponseChan is closed
			select {
			case msg, ok := <-stream.ResponseChan():
				require.False(t, ok, "Expected ResponseChan to be closed, got message: %v", msg)
			case <-time.After(time.Second):
				t.Fatal("Timeout waiting for stream to close")
			}

			// Verify calling Stop() again is safe (should be idempotent)
			stream.Stop()

			// Double-check the channel remains closed
			_, ok := <-stream.ResponseChan()
			require.False(t, ok, "Channel should remain closed after multiple Stop calls")
		})
	})

	t.Run("Concurrent Requests", func(t *testing.T) {
		ctx := context.Background()
		messages := []Message{
			{Role: "user", Content: "Count to 3 and nothing else"},
		}

		results := make(chan string, 3)
		errors := make(chan error, 3)

		for i := 0; i < 3; i++ {
			go func() {
				stream, err := client.GenerateStream(ctx, messages)
				if err != nil {
					errors <- err
					results <- ""
					return
				}

				var response string
				for msg := range stream.ResponseChan() {
					if msg.Error != nil {
						errors <- msg.Error
						results <- ""
						return
					}
					response += msg.Content
				}
				errors <- nil
				results <- response
			}()
		}

		for i := 0; i < 3; i++ {
			err := <-errors
			require.NoError(t, err)

			response := <-results
			require.NotEmpty(t, response)
			require.Contains(t, response, "1")
			require.Contains(t, response, "2")
			require.Contains(t, response, "3")
			t.Logf("Concurrent Response %d: %s", i+1, response)
		}
	})

	t.Run("APIModelManager Integration", func(t *testing.T) {
		manager, err := NewAPIModelManager()
		require.NoError(t, err)

		gpt4ID := APIModelToDigest(APIModelIDGPT4)
		model, err := manager.GetAPIModel(gpt4ID)
		require.NoError(t, err)
		require.NotNil(t, model)

		ctx := context.Background()
		messages := []Message{
			{Role: "user", Content: "Say 'Test successful' and nothing else"},
		}

		stream, err := model.GenerateStream(ctx, messages)
		require.NoError(t, err)

		var response string
		for msg := range stream.ResponseChan() {
			require.NoError(t, msg.Error)
			response += msg.Content
		}

		require.Contains(t, response, "Test successful")
	})
}

func TestModelIDConversion(t *testing.T) {
	testCases := []struct {
		name    string
		modelID byte
		wantID  byte
	}{
		{"GPT-4", APIModelIDGPT4, 1},
		{"GPT-3.5", APIModelIDGPT35Turbo, 2},
		{"Claude-3 Opus", APIModelIDClaude3Opus, 3},
		{"Claude-3 Sonnet", APIModelIDClaude3Sonnet, 4},
		{"Claude-3 Haiku", APIModelIDClaude3Haiku, 5},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			digest := APIModelToDigest(tc.modelID)

			// Check the last byte
			require.Equal(t, tc.wantID, digest.Bytes()[31])

			// Verify all other bytes are zero
			for i := 0; i < 31; i++ {
				require.Equal(t, byte(0), digest.Bytes()[i])
			}
		})
	}
}
