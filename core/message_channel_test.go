package core

import (
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/stretchr/testify/require"
	"testing"
	"time"
)

func TestMessageChannel(t *testing.T) {
	t.Run("EnqueueRequest and GetRequestChannel", func(t *testing.T) {
		mc := NewMessageChannel(100, 100)

		// Create test request
		reqID := db.NewDigest([]byte("test-request"))
		connID := "test-connection"

		req, err := NewWrappedRequest[*InferenceRequest](reqID, &InferenceRequest{
			UserID:   db.NewDigest([]byte("123")),
			ModelID:  "gpt-4",
			Messages: []Message{},
		})
		require.NoError(t, err)

		// Test enqueue
		err = mc.EnqueueRequest(req, connID)
		require.NoError(t, err)

		// Test receiving from channel
		select {
		case req := <-mc.GetRequestChannel():
			require.Equal(t, reqID, req.RequestID)
			require.Equal(t, connID, req.ConnectionID)
		case <-time.After(time.Second):
			t.Fatal("Timeout waiting for request")
		}
	})

	t.Run("EnqueueResponse and GetResponseChannel", func(t *testing.T) {
		mc := NewMessageChannel(100, 100)

		// Create test response
		reqID := db.NewDigest([]byte("test-request"))
		connID := "test-connection"

		resp, err := NewWrappedResponse[*InferenceResponse](reqID, &InferenceResponse{
			Type:    ResponseTypeFinal,
			Content: "Hello world",
			Status:  ResponseStatusSuccess,
		})
		require.NoError(t, err)

		// Start goroutine to receive response
		done := make(chan struct{})
		go func() {
			respChan := mc.GetResponseChannel()
			received := <-respChan
			require.Equal(t, resp, received.Response)
			require.Equal(t, connID, received.ConnectionID)
			close(done)
		}()

		// Send response
		mc.EnqueueResponse(resp, connID)

		// Wait for response to be received
		select {
		case <-done:
			// Test passed
		case <-time.After(time.Second):
			t.Fatal("Timeout waiting for response")
		}
	})

	t.Run("Channel Full Behavior", func(t *testing.T) {
		mc := NewMessageChannel(2, 2)

		// Fill up the channel
		for i := 0; i < 2; i++ {
			reqID := db.NewDigest([]byte(fmt.Sprintf("test-request-%d", i)))
			req, err := NewWrappedRequest[*InferenceRequest](reqID, &InferenceRequest{
				UserID:   db.NewDigest([]byte(fmt.Sprintf("user-%d", i))),
				ModelID:  "gpt-4",
				Messages: []Message{},
			})
			require.NoError(t, err)
			err = mc.EnqueueRequest(req, "conn-1")
			require.NoError(t, err)
		}

		// Try to enqueue one more request
		reqID := db.NewDigest([]byte("overflow-request"))
		req, err := NewWrappedRequest[*InferenceRequest](reqID, &InferenceRequest{
			UserID:   db.NewDigest([]byte("overflow-user")),
			ModelID:  "gpt-4",
			Messages: []Message{},
		})
		require.NoError(t, err)
		err = mc.EnqueueRequest(req, "conn-1")
		require.Error(t, err)
		require.Contains(t, err.Error(), "channel full")
	})

	t.Run("Channel Close", func(t *testing.T) {
		mc := NewMessageChannel(100, 100)

		// Send a request
		reqID := db.NewDigest([]byte("test-request"))
		req, err := NewWrappedRequest[*InferenceRequest](reqID, &InferenceRequest{
			UserID:   db.NewDigest([]byte("test-user")),
			ModelID:  "gpt-4",
			Messages: []Message{},
		})
		require.NoError(t, err)
		err = mc.EnqueueRequest(req, "conn-1")
		require.NoError(t, err)

		// Close channels
		mc.Close()

		// Verify channels are closed
		_, ok := <-mc.GetRequestChannel()
		require.True(t, ok, "request channel should allow reading of the pending items")
		_, ok = <-mc.GetRequestChannel()
		require.False(t, ok, "request channel should now be closed")

		_, ok = <-mc.GetResponseChannel()
		require.False(t, ok, "response channel should be closed")

		// Verify we can't enqueue after close
		err = mc.EnqueueRequest(req, "conn-1")
		require.Error(t, err)
	})

	t.Run("Response Timestamp", func(t *testing.T) {
		mc := NewMessageChannel(100, 100)

		before := time.Now()
		time.Sleep(time.Millisecond) // Ensure measurable time difference

		reqID := db.NewDigest([]byte("test-request"))
		resp, err := NewWrappedResponse[*InferenceResponse](reqID, &InferenceResponse{
			Type:    ResponseTypeFinal,
			Content: "Test response",
			Status:  ResponseStatusSuccess,
		})
		require.NoError(t, err)
		mc.EnqueueResponse(resp, "conn-1")

		response := <-mc.GetResponseChannel()

		require.True(t, response.Timestamp.After(before), "response timestamp should be after test start")
		require.True(t, response.Timestamp.Before(time.Now()), "response timestamp should be before now")
	})
}
