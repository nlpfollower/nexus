package core

import (
	"encoding/json"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"io"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

type ConnectionManager struct {
	port        int
	messageChan *MessageChannel
	listener    net.Listener
	conns       *utils.ConcurrentMap[string, *Connection]
	nextConnID  atomic.Uint64

	// Callback with connectionID when a connection is closed
	onConnectionClosed func(connectionID string)
}

type Connection struct {
	ID      string
	Conn    net.Conn
	Encoder *json.Encoder
	mu      sync.Mutex // protects Encoder
}

func NewConnectionManager(port int, queue *MessageChannel, onClosed func(string)) *ConnectionManager {
	return &ConnectionManager{
		port:               port,
		messageChan:        queue,
		conns:              utils.NewConcurrentMap[string, *Connection](),
		onConnectionClosed: onClosed,
	}
}

func (cm *ConnectionManager) Start() error {
	var err error
	cm.listener, err = net.Listen("tcp", fmt.Sprintf(":%d", cm.port))
	if err != nil {
		return fmt.Errorf("failed to start listener: %w", err)
	}

	log.Printf("Connection manager listening on port %d", cm.port)

	// Accept connections in a goroutine
	go cm.connectionsLoop()

	// Handle responses in a goroutine
	go cm.responseLoop()

	return nil
}

func (cm *ConnectionManager) Stop() error {
	if cm.listener != nil {
		if err := cm.listener.Close(); err != nil {
			return fmt.Errorf("failed to close listener: %w", err)
		}
	}

	// Close all active connections
	conns := cm.conns.ToMap()
	for _, conn := range conns {
		conn.Conn.Close()
	}

	return nil
}

func (cm *ConnectionManager) connectionsLoop() {
	for {
		netConn, err := cm.listener.Accept()
		if err != nil {
			if ne, ok := err.(net.Error); ok && ne.Temporary() {
				log.Printf("Temporary accept error: %v", err)
				time.Sleep(time.Second)
				continue
			}
			// If listener is closed, just return
			return
		}

		connID := cm.generateConnectionID()
		conn := &Connection{
			ID:      connID,
			Conn:    netConn,
			Encoder: json.NewEncoder(netConn),
		}

		cm.conns.Set(connID, conn)
		go cm.handleConnection(conn)
	}
}

func (cm *ConnectionManager) handleConnection(conn *Connection) {
	defer func() {
		if cm.onConnectionClosed != nil {
			cm.onConnectionClosed(conn.ID)
		}
		conn.Conn.Close()
		cm.conns.Remove(conn.ID)
		log.Printf("Connection %s closed", conn.ID)
	}()

	decoder := json.NewDecoder(conn.Conn)

	for {
		var wrapped WrappedRequest
		if err := decoder.Decode(&wrapped); err != nil {
			if err != io.EOF {
				log.Printf("Error reading from connection %s: %v", conn.ID, err)
			}
			return
		}

		if err := cm.messageChan.EnqueueRequest(&wrapped, conn.ID); err != nil {
			log.Printf("Error enqueueing request from connection %s: %v", conn.ID, err)
			cm.sendErrorResponse(conn, wrapped.RequestID, err)
			continue
		}
	}
}

func (cm *ConnectionManager) responseLoop() {
	responseChan := cm.messageChan.GetResponseChannel()

	for resp := range responseChan {
		if err := cm.sendResponse(resp.ConnectionID, resp.Response); err != nil {
			log.Printf("Failed to send response to connection %s: %v", resp.ConnectionID, err)
		}
	}
}

func (cm *ConnectionManager) sendResponse(connID string, response *WrappedResponse) error {
	conn, ok := cm.conns.Get(connID)
	if !ok {
		return fmt.Errorf("connection %s not found", connID)
	}

	conn.mu.Lock()
	defer conn.mu.Unlock()

	return conn.Encoder.Encode(response)
}

func (cm *ConnectionManager) sendErrorResponse(conn *Connection, requestID db.Digest, err error) {
	resp := &WrappedResponse{
		RequestID: requestID,
		Type:      "ERROR",
		Data:      json.RawMessage(fmt.Sprintf(`{"error":"%s"}`, err.Error())),
	}

	conn.mu.Lock()
	defer conn.mu.Unlock()

	if err := conn.Encoder.Encode(resp); err != nil {
		log.Printf("Failed to send error response to connection %s: %v", conn.ID, err)
	}
}

func (cm *ConnectionManager) generateConnectionID() string {
	id := cm.nextConnID.Add(1)
	return fmt.Sprintf("conn_%d", id)
}
