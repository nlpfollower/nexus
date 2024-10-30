package core

import (
	"encoding/json"
	"fmt"
	"github.com/joho/godotenv"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/stretchr/testify/require"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

// Helper to set up a test nexus
func setupTestNexus(t *testing.T, cfg *Config) *Nexus {
	nexus, err := NewNexus(cfg)
	require.NoError(t, err)

	// Start the nexus
	err = nexus.Start()
	require.NoError(t, err)

	return nexus
}

func TestNexus(t *testing.T) {
	// Skip in CI environment
	if os.Getenv("CI") == "true" {
		t.Skip("Skipping integration test in CI environment")
	}

	// Verify API key is available
	apiKey := os.Getenv("GPT_API_KEY")
	require.NotEmpty(t, apiKey, "GPT_API_KEY environment variable is required")

	// Get an available port for testing
	listener, err := net.Listen("tcp", ":0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	cfg := &Config{
		Port: port,
	}

	t.Run("API Model Inference", func(t *testing.T) {
		nexus := setupTestNexus(t, cfg)

		// Create connection before setting up defer
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)

		// Setup cleanup after connection is established
		cleanup := func() {
			conn.Close()
			nexus.Stop()
		}
		defer cleanup()

		// Setup request
		userID := db.NewDigest([]byte("test-user"))
		modelID := APIModelToDigest(APIModelIDGPT4)
		reqID := db.NewDigest([]byte("test-request"))

		inferReq := &InferenceRequest{
			UserID:  userID,
			ModelID: modelID,
			Messages: []Message{
				{Role: "system", Content: "You are a helpful assistant. Keep responses very short."},
				{Role: "user", Content: "What is 2+2?"},
			},
		}

		wrapped, err := NewWrappedRequest(reqID, inferReq)
		require.NoError(t, err)

		err = json.NewEncoder(conn).Encode(wrapped)
		require.NoError(t, err)

		var responses []WrappedResponse
		var fullResponse string
		done := make(chan struct{})
		decoderErr := make(chan error, 1)

		// Start reading responses in background
		go func() {
			decoder := json.NewDecoder(conn)
			for {
				var resp WrappedResponse
				if err := decoder.Decode(&resp); err != nil {
					if !isConnectionClosed(err) {
						decoderErr <- err
					}
					close(done)
					return
				}

				var inferResp InferenceResponse
				if err := json.Unmarshal(resp.Data, &inferResp); err != nil {
					decoderErr <- err
					close(done)
					return
				}

				responses = append(responses, resp)
				fullResponse += inferResp.Content

				fmt.Println("Response:", inferResp.Content, "Type:", inferResp.Type)
				if inferResp.Type == ResponseTypeFinal {
					close(done)
					return
				}
			}
		}()

		// Wait for completion or timeout
		timeoutCh := time.After(5 * time.Second)
		select {
		case <-timeoutCh:
			cleanup() // Ensure cleanup runs before the test fails
			t.Fatal("Timeout waiting for responses")
		case err := <-decoderErr:
			cleanup()
			t.Fatal("Error reading response:", err)
		case <-done:
			// Success path - continue with verification
		}

		// Verify responses
		require.NotEmpty(t, responses, "No responses received")
		require.Equal(t, reqID, responses[0].RequestID)

		var firstResp InferenceResponse
		err = json.Unmarshal(responses[0].Data, &firstResp)
		require.NoError(t, err)
		require.Equal(t, ResponseTypePartial, firstResp.Type)
		require.Equal(t, ResponseStatusSuccess, firstResp.Status)
		require.NotEmpty(t, firstResp.Content)

		var lastResp InferenceResponse
		err = json.Unmarshal(responses[len(responses)-1].Data, &lastResp)
		require.NoError(t, err)
		require.Equal(t, ResponseTypeFinal, lastResp.Type)
		require.Equal(t, ResponseStatusSuccess, lastResp.Status)

		require.Contains(t, fullResponse, "4")
		t.Logf("Full response: %s", fullResponse)
	})

	t.Run("Set User Request", func(t *testing.T) {
		nexus := setupTestNexus(t, cfg)
		defer nexus.Stop()

		// Create a real connection to the nexus
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)
		defer conn.Close()

		userID := db.NewDigest([]byte("test-user"))
		reqID := db.NewDigest([]byte("test-request"))

		userReq := &SetUserRequest{
			UserID: userID,
		}

		// Create and send wrapped request
		wrapped, err := NewWrappedRequest(reqID, userReq)
		require.NoError(t, err)

		err = json.NewEncoder(conn).Encode(wrapped)
		require.NoError(t, err)

		// Read response
		var resp WrappedResponse
		err = json.NewDecoder(conn).Decode(&resp)
		require.NoError(t, err)
		require.Equal(t, reqID, resp.RequestID)

		var userResp SetUserResponse
		err = json.Unmarshal(resp.Data, &userResp)
		require.NoError(t, err)
		require.Equal(t, ResponseStatusSuccess, userResp.Status)

		// Verify user was tracked
		require.True(t, nexus.tracker.UserExists(userID))
	})

	t.Run("Error Handling", func(t *testing.T) {
		nexus := setupTestNexus(t, cfg)
		defer nexus.Stop()

		// Create a real connection to the nexus
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)
		defer conn.Close()

		reqID := db.NewDigest([]byte("test-request"))

		inferReq := &InferenceRequest{
			UserID:   db.NewDigest([]byte("test-user")),
			ModelID:  db.NewDigest([]byte("invalid-model")),
			Messages: []Message{{Role: "user", Content: "test"}},
		}

		// Create and send wrapped request
		wrapped, err := NewWrappedRequest(reqID, inferReq)
		require.NoError(t, err)

		err = json.NewEncoder(conn).Encode(wrapped)
		require.NoError(t, err)

		// Read response
		var resp WrappedResponse
		err = json.NewDecoder(conn).Decode(&resp)
		require.NoError(t, err)
		require.Equal(t, reqID, resp.RequestID)

		var inferResp InferenceResponse
		err = json.Unmarshal(resp.Data, &inferResp)
		require.NoError(t, err)
		require.Equal(t, ResponseStatusError, inferResp.Status)
		require.Contains(t, inferResp.Content, "infrastructure sessions not yet implemented")
	})
}

func TestGenerationCancellation(t *testing.T) {
	// Get an available port
	listener, err := net.Listen("tcp", ":0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	cfg := &Config{Port: port}
	nexus := setupTestNexus(t, cfg)
	defer nexus.Stop()

	// Helper function to wait for generation response
	waitForGenerationStart := func(t *testing.T, decoder *json.Decoder) {
		t.Helper()
		var resp WrappedResponse
		err := decoder.Decode(&resp)
		require.NoError(t, err)

		var inferResp InferenceResponse
		err = json.Unmarshal(resp.Data, &inferResp)
		require.NoError(t, err)
		require.Equal(t, ResponseTypePartial, inferResp.Type)
		require.NotEmpty(t, inferResp.Content)
	}

	t.Run("Client Disconnect During Generation", func(t *testing.T) {
		// Create connection
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)

		// Send inference request
		userID := db.NewDigest([]byte("test-user"))
		modelID := APIModelToDigest(APIModelIDGPT4)
		reqID := db.NewDigest([]byte("test-request"))

		inferReq := &InferenceRequest{
			UserID:  userID,
			ModelID: modelID,
			Messages: []Message{
				{Role: "user", Content: "Write a short story about a dog"},
			},
		}

		wrapped, err := NewWrappedRequest(reqID, inferReq)
		require.NoError(t, err)

		err = json.NewEncoder(conn).Encode(wrapped)
		require.NoError(t, err)

		// Wait for first response to ensure generation started
		decoder := json.NewDecoder(conn)
		waitForGenerationStart(t, decoder)

		// Verify generation is tracked
		require.Eventually(t, func() bool {
			_, exists := nexus.activeGenerations.Get(reqID.String())
			return exists
		}, time.Second, 50*time.Millisecond)

		// Close connection mid-generation
		conn.Close()

		// Verify generation is cleaned up
		require.Eventually(t, func() bool {
			_, exists := nexus.activeGenerations.Get(reqID.String())
			return !exists
		}, time.Second, 50*time.Millisecond)
	})

	t.Run("Multiple Concurrent Generations", func(t *testing.T) {
		numConns := 3
		conns := make([]net.Conn, numConns)
		reqIDs := make([]db.Digest, numConns)
		decoders := make([]*json.Decoder, numConns)

		// Start multiple generations
		for i := 0; i < numConns; i++ {
			conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
			require.NoError(t, err)
			conns[i] = conn
			decoders[i] = json.NewDecoder(conn)

			reqIDs[i] = db.NewDigest([]byte(fmt.Sprintf("test-request-%d", i)))
			inferReq := &InferenceRequest{
				UserID:  db.NewDigest([]byte("test-user")),
				ModelID: APIModelToDigest(APIModelIDGPT4),
				Messages: []Message{
					{Role: "user", Content: "Write a short story about a cat"},
				},
			}

			wrapped, err := NewWrappedRequest(reqIDs[i], inferReq)
			require.NoError(t, err)

			err = json.NewEncoder(conn).Encode(wrapped)
			require.NoError(t, err)

			// Wait for generation to start
			waitForGenerationStart(t, decoders[i])
		}

		// Wait for all generations to be tracked
		require.Eventually(t, func() bool {
			count := 0
			for i := 0; i < numConns; i++ {
				if _, exists := nexus.activeGenerations.Get(reqIDs[i].String()); exists {
					count++
				}
			}
			return count == numConns
		}, 5*time.Second, 100*time.Millisecond)

		// Close connections in random order
		var wg sync.WaitGroup
		for i := 0; i < numConns; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				time.Sleep(time.Duration(50*idx) * time.Millisecond)
				conns[idx].Close()
			}(i)
		}

		// Wait for all connections to close
		wg.Wait()

		// Verify all generations are cleaned up
		require.Eventually(t, func() bool {
			return len(nexus.activeGenerations.GetAll()) == 0
		}, 5*time.Second, 100*time.Millisecond)
	})

	t.Run("New Requests After Disconnect", func(t *testing.T) {
		// First connection and request
		conn1, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)

		reqID1 := db.NewDigest([]byte("test-request-1"))
		inferReq := &InferenceRequest{
			UserID:  db.NewDigest([]byte("test-user")),
			ModelID: APIModelToDigest(APIModelIDGPT4),
			Messages: []Message{
				{Role: "user", Content: "Write a short story about a cat"},
			},
		}

		wrapped, err := NewWrappedRequest(reqID1, inferReq)
		require.NoError(t, err)

		err = json.NewEncoder(conn1).Encode(wrapped)
		require.NoError(t, err)

		// Wait for first generation to start
		decoder1 := json.NewDecoder(conn1)
		waitForGenerationStart(t, decoder1)

		// Verify first generation is tracked
		require.Eventually(t, func() bool {
			_, exists := nexus.activeGenerations.Get(reqID1.String())
			return exists
		}, 5*time.Second, 50*time.Millisecond)

		// Close first connection
		conn1.Close()

		// Verify cleanup of first generation
		require.Eventually(t, func() bool {
			_, exists := nexus.activeGenerations.Get(reqID1.String())
			return !exists
		}, 5*time.Second, 50*time.Millisecond)

		// Create new connection and request
		conn2, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)
		defer conn2.Close()

		reqID2 := db.NewDigest([]byte("test-request-2"))
		wrapped, err = NewWrappedRequest(reqID2, inferReq)
		require.NoError(t, err)

		err = json.NewEncoder(conn2).Encode(wrapped)
		require.NoError(t, err)

		// Verify new request works
		decoder2 := json.NewDecoder(conn2)
		waitForGenerationStart(t, decoder2)

		// Verify second generation is tracked
		require.Eventually(t, func() bool {
			_, exists := nexus.activeGenerations.Get(reqID2.String())
			return exists
		}, 5*time.Second, 50*time.Millisecond)
	})
}

func isConnectionClosed(err error) bool {
	return err.Error() == "EOF" ||
		strings.Contains(err.Error(), "use of closed network connection") ||
		strings.Contains(err.Error(), "connection reset by peer")
}
