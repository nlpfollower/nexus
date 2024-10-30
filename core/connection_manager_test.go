package core

import (
	"encoding/json"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/stretchr/testify/require"
	"net"
	"sync/atomic"
	"testing"
	"time"
)

func setupTestManager(t *testing.T) (*ConnectionManager, *MessageChannel, *atomic.Int32, int) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	messageChan := NewMessageChannel(100, 100)
	closedConnections := &atomic.Int32{}

	manager := NewConnectionManager(port, messageChan, func(connID string) {
		closedConnections.Add(1)
	})

	err = manager.Start()
	require.NoError(t, err)

	return manager, messageChan, closedConnections, port
}

func TestConnectionManager(t *testing.T) {
	t.Run("Single Connection", func(t *testing.T) {
		manager, messageChan, _, port := setupTestManager(t)
		defer manager.Stop()
		requestChan := messageChan.GetRequestChannel()

		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)
		defer conn.Close()

		requestID := db.NewDigest([]byte("test-request"))
		inferReq := &InferenceRequest{
			UserID:   db.NewDigest([]byte("123")),
			ModelID:  APIModelToDigest(APIModelIDGPT4),
			Messages: []Message{},
		}

		request, err := NewWrappedRequest[*InferenceRequest](requestID, inferReq)
		require.NoError(t, err)

		encoder := json.NewEncoder(conn)
		err = encoder.Encode(request)
		require.NoError(t, err)

		require.Equal(t, 1, manager.conns.Count())

		select {
		case req := <-requestChan:
			require.Equal(t, requestID, req.RequestID)

			resp := &InferenceResponse{
				Type:    ResponseTypePartial,
				Content: "test response",
				Status:  ResponseStatusSuccess,
			}

			response, err := NewWrappedResponse[*InferenceResponse](requestID, resp)
			require.NoError(t, err)

			messageChan.EnqueueResponse(response, req.ConnectionID)

			decoder := json.NewDecoder(conn)
			var received WrappedResponse
			err = decoder.Decode(&received)
			require.NoError(t, err)
			require.Equal(t, requestID, received.RequestID)

		case <-time.After(time.Second):
			t.Fatal("Timeout waiting for request")
		}
	})

	t.Run("Multiple Connections", func(t *testing.T) {
		manager, messageChan, _, port := setupTestManager(t)
		defer manager.Stop()
		requestChan := messageChan.GetRequestChannel()

		numConns := 3
		conns := make([]net.Conn, numConns)
		decoders := make([]*json.Decoder, numConns)
		requestIDs := make([]db.Digest, numConns)

		for i := 0; i < numConns; i++ {
			conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
			require.NoError(t, err)
			defer conn.Close()

			conns[i] = conn
			decoders[i] = json.NewDecoder(conn)
			requestIDs[i] = db.NewDigest([]byte(fmt.Sprintf("test-request-%d", i)))

			inferReq := &InferenceRequest{
				UserID:   db.NewDigest([]byte("123")),
				ModelID:  APIModelToDigest(APIModelIDGPT4),
				Messages: []Message{},
			}

			request, err := NewWrappedRequest[*InferenceRequest](requestIDs[i], inferReq)
			require.NoError(t, err)

			err = json.NewEncoder(conn).Encode(request)
			require.NoError(t, err)
		}

		require.Eventually(t, func() bool {
			return manager.conns.Count() == numConns
		}, time.Second, 50*time.Millisecond)

		receivedReqs := make([]*Request, 0, numConns)
		for i := 0; i < numConns; i++ {
			select {
			case req := <-requestChan:
				receivedReqs = append(receivedReqs, req)

				resp := &InferenceResponse{
					Type:    ResponseTypePartial,
					Content: "test response",
					Status:  ResponseStatusSuccess,
				}

				response, err := NewWrappedResponse[*InferenceResponse](req.RequestID, resp)
				require.NoError(t, err)
				messageChan.EnqueueResponse(response, req.ConnectionID)

			case <-time.After(time.Second):
				t.Fatal("Timeout waiting for request")
			}
		}

		require.Len(t, receivedReqs, numConns)

		for i := 0; i < numConns; i++ {
			var received WrappedResponse
			err := decoders[i].Decode(&received)
			require.NoError(t, err)
			require.Contains(t, requestIDs, received.RequestID)
		}
	})

	t.Run("Connection Close", func(t *testing.T) {
		manager, messageChan, closedConnections, port := setupTestManager(t)
		defer manager.Stop()
		requestChan := messageChan.GetRequestChannel()

		initialClosedCount := closedConnections.Load()
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)

		requestID := db.NewDigest([]byte("test-close"))
		inferReq := &InferenceRequest{
			UserID:   db.NewDigest([]byte("test-user")),
			ModelID:  APIModelToDigest(APIModelIDGPT4),
			Messages: []Message{},
		}

		request, err := NewWrappedRequest[*InferenceRequest](requestID, inferReq)
		require.NoError(t, err)

		err = json.NewEncoder(conn).Encode(request)
		require.NoError(t, err)

		var connID string
		select {
		case req := <-requestChan:
			connID = req.ConnectionID
		case <-time.After(time.Second):
			t.Fatal("Timeout waiting for request")
		}

		require.Equal(t, 1, manager.conns.Count())

		conn.Close()

		require.Eventually(t, func() bool {
			_, exists := manager.conns.Get(connID)
			return !exists && closedConnections.Load() > initialClosedCount
		}, time.Second, 50*time.Millisecond)

		require.Equal(t, 0, manager.conns.Count())
	})

	t.Run("Shutdown", func(t *testing.T) {
		manager, messageChan, closedConnections, port := setupTestManager(t)
		defer manager.Stop()
		requestChan := messageChan.GetRequestChannel()

		initialClosedCount := closedConnections.Load()

		numConns := 3
		conns := make([]net.Conn, numConns)
		for i := 0; i < numConns; i++ {
			conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
			require.NoError(t, err)
			conns[i] = conn

			requestID := db.NewDigest([]byte(fmt.Sprintf("test-shutdown-%d", i)))
			inferReq := &InferenceRequest{
				UserID:   db.NewDigest([]byte("test-user")),
				ModelID:  APIModelToDigest(APIModelIDGPT4),
				Messages: []Message{},
			}
			request, err := NewWrappedRequest[*InferenceRequest](requestID, inferReq)
			require.NoError(t, err)
			err = json.NewEncoder(conn).Encode(request)
			require.NoError(t, err)
		}

		require.Eventually(t, func() bool {
			return manager.conns.Count() == numConns
		}, time.Second, 50*time.Millisecond)

		for i := 0; i < numConns; i++ {
			select {
			case <-requestChan:
				// Request received successfully
			case <-time.After(time.Second):
				t.Fatal("Timeout waiting for request")
			}
		}

		err := manager.Stop()
		require.NoError(t, err)

		require.Eventually(t, func() bool {
			return closedConnections.Load() >= initialClosedCount+int32(numConns)
		}, time.Second, 50*time.Millisecond)

		require.Eventually(t, func() bool {
			return manager.conns.Count() == 0
		}, time.Second, 50*time.Millisecond)

		_, err = net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.Error(t, err)
	})

	t.Run("Connection Error Handling2", func(t *testing.T) {
		manager, _, closedConnections, port := setupTestManager(t)
		defer manager.Stop()

		initialClosedCount := closedConnections.Load()
		conn, err := net.Dial("tcp", fmt.Sprintf(":%d", port))
		require.NoError(t, err)

		_, err = conn.Write([]byte("invalid json\n"))
		require.NoError(t, err)

		require.Eventually(t, func() bool {
			return closedConnections.Load() > initialClosedCount && manager.conns.Count() == 0
		}, time.Second, 50*time.Millisecond)
	})
}
