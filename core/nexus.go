// core/nexus.go
package core

import (
	"context"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"log"
	"sync"
)

type activeGeneration struct {
	stream       GenerationStream
	requestID    db.Digest
	connectionID string
}

type Config struct {
	Port int
}

// Update the Nexus struct
type Nexus struct {
	queue       *MessageChannel
	tracker     *RequestTracker
	apiManager  *APIModelManager
	sessionMgr  *SessionManager
	connManager *ConnectionManager

	// Add generation tracking
	// key is requestID.String()
	activeGenerations *utils.ConcurrentMap[string, *activeGeneration]

	// Channels for coordination
	sessionActiveCh  chan string
	directResponseCh chan *Request

	wg        sync.WaitGroup
	done      chan struct{}
	closeOnce sync.Once
}

func NewNexus(cfg *Config) (*Nexus, error) {
	apiManager, err := NewAPIModelManager()
	if err != nil {
		return nil, fmt.Errorf("failed to create API model manager: %w", err)
	}

	queue := NewMessageChannel(1000, 1000)

	nexus := &Nexus{
		queue:             queue,
		tracker:           NewRequestTracker(),
		apiManager:        apiManager,
		sessionMgr:        NewSessionManager(),
		activeGenerations: utils.NewConcurrentMap[string, *activeGeneration](),
		sessionActiveCh:   make(chan string, 100),
		directResponseCh:  make(chan *Request, 1000),
		done:              make(chan struct{}),
	}

	// Create connection manager with closure callback
	nexus.connManager = NewConnectionManager(
		cfg.Port,
		queue,
		nexus.handleConnectionClosed,
	)

	return nexus, nil
}

func (n *Nexus) Start() error {
	if err := n.connManager.Start(); err != nil {
		return fmt.Errorf("failed to start connection manager: %w", err)
	}

	// Start request processing loop
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		n.requestLoop()
	}()

	// Start response handling loop
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		n.responseLoop()
	}()

	return nil
}

func (n *Nexus) Stop() {
	n.closeOnce.Do(func() {
		// First signal all goroutines to stop
		close(n.done)

		// Stop accepting new connections first
		if err := n.connManager.Stop(); err != nil {
			log.Printf("Error stopping connection manager: %v", err)
		}

		// Stop all active generations
		for _, gen := range n.activeGenerations.GetAll() {
			gen.stream.Stop()
		}

		// Wait for all goroutines to finish
		n.wg.Wait()

		// Now it's safe to close channels
		close(n.sessionActiveCh)
		close(n.directResponseCh)
		n.queue.Close()
	})
}

func (n *Nexus) stopGeneration(requestID db.Digest) {
	genKey := requestID.String()
	if gen, exists := n.activeGenerations.Get(genKey); exists {
		gen.stream.Stop()
		n.activeGenerations.Remove(genKey)
		log.Printf("Stopped generation %s", genKey)
	}
}

func (n *Nexus) handleConnectionClosed(connectionID string) {
	var wg sync.WaitGroup
	gens := n.activeGenerations.GetAll()
	for _, gen := range gens {
		if gen.connectionID == connectionID {
			wg.Add(1)
			go func(g *activeGeneration) {
				defer wg.Done()
				g.stream.Stop()
				n.activeGenerations.Remove(g.requestID.String())
			}(gen)
		}
	}
	wg.Wait() // Wait for all generations to be cleaned up
	log.Printf("Cleaned up all generations for connection %s", connectionID)
}

func (n *Nexus) sendWrappedResponse(req *Request, resp NexusResponse) {
	wrapped, err := NewWrappedResponse(req.RequestID, resp)
	if err != nil {
		n.sendErrorResponse(req, err)
		return
	}
	n.queue.EnqueueResponse(wrapped, req.ConnectionID)
}

// ====================
// Request handling
// ====================
func (n *Nexus) requestLoop() {
	requestChan := n.queue.GetRequestChannel()

	for {
		select {
		case <-n.done:
			return
		case req, ok := <-requestChan:
			if !ok {
				return
			}
			if err := n.processRequest(req); err != nil {
				log.Printf("Error processing request: %v", err)
			}
		}
	}
}

func (n *Nexus) processRequest(req *Request) error {
	n.tracker.AddRequest(req)

	switch req.Type {
	case RequestTypeInference:
		if err := n.routeInferenceRequest(req); err != nil {
			log.Printf("Error routing inference request %s: %v", req.RequestID, err)
			n.sendErrorResponse(req, err)
		}
	case RequestTypeSetUser:
		// User requests can be handled immediately
		n.directResponseCh <- req
	default:
		err := fmt.Errorf("unknown request type: %s", req.Type)
		log.Printf("Error processing request: %v", err)
		n.sendErrorResponse(req, err)
	}

	return nil
}

func (n *Nexus) routeInferenceRequest(req *Request) error {
	inferReq, ok := req.Data.(*InferenceRequest)
	if !ok {
		return fmt.Errorf("invalid inference request data")
	}

	// Check if this is an API model
	if n.apiManager.IsAPIModel(inferReq.ModelID) {
		n.directResponseCh <- req
		return nil
	}

	// Handle infrastructure-based model request
	session, err := n.sessionMgr.GetOrCreateSession(inferReq.ModelID)
	if err != nil {
		return fmt.Errorf("failed to get/create session: %w", err)
	}

	if session.Status == SessionStatusRunning {
		n.sessionActiveCh <- session.ID
	}

	return nil
}

// ====================
// Response handling
// ====================
func (n *Nexus) responseLoop() {
	for {
		select {
		case <-n.done:
			return
		case sessionID := <-n.sessionActiveCh:
			if session, ok := n.sessionMgr.GetSession(sessionID); ok {
				go n.handleSessionProcessing(session)
			}
		case req := <-n.directResponseCh:
			go n.handleDirectRequest(req)
		}
	}
}

func (n *Nexus) handleDirectRequest(req *Request) {
	var err error
	switch req.Type {
	case RequestTypeInference:
		err = n.handleAPIInference(req)
	case RequestTypeSetUser:
		err = n.handleSetUser(req)
	default:
		err = fmt.Errorf("unknown request type: %s", req.Type)
	}

	if err != nil {
		n.sendErrorResponse(req, err)
	}
}

// Inference response handling
func (n *Nexus) handleSessionProcessing(session *Session) {
	// TODO: Implementation for infrastructure-based processing
	// This will:
	// 1. Collect requests into batches
	// 2. Process batches through infrastructure
	// 3. Stream responses back to clients
	log.Printf("Session %s ready for processing (not yet implemented)", session.ID)
}

func (n *Nexus) handleAPIInference(req *Request) error {
	inferReq, ok := req.Data.(*InferenceRequest)
	if !ok {
		return fmt.Errorf("invalid inference request data")
	}

	model, err := n.apiManager.GetAPIModel(inferReq.ModelID)
	if err != nil {
		return fmt.Errorf("failed to get API model: %w", err)
	}
	fmt.Println("handleAPIInference messages:", inferReq.Messages)

	ctx := context.Background()
	stream, err := model.GenerateStream(ctx, inferReq.Messages)
	if err != nil {
		return fmt.Errorf("failed to start inference: %w", err)
	}

	// Track the active generation
	n.activeGenerations.Set(req.RequestID.String(), &activeGeneration{
		stream:       stream,
		requestID:    req.RequestID,
		connectionID: req.ConnectionID,
	})

	// Start handling the stream
	go n.handleInferenceStream(req, stream)
	return nil
}

func (n *Nexus) handleInferenceStream(req *Request, stream GenerationStream) {
	key := req.RequestID.String()
	defer func() {
		stream.Stop()
		n.activeGenerations.Remove(key)
		log.Printf("Removed generation %s from active generations", key)
	}()

	// Ensure the generation is tracked before reading responses
	if _, exists := n.activeGenerations.Get(key); !exists {
		return
	}

	responseChan := stream.ResponseChan()
	for {
		select {
		case <-n.done:
			return
		case resp, ok := <-responseChan:
			if !ok {
				n.sendFinalResponse(req)
				return
			}
			if resp.Error != nil {
				n.sendErrorResponse(req, resp.Error)
				return
			}
			if resp.Content != "" {
				n.sendPartialResponse(req, resp.Content)
			}
		}
	}
}

// Helper methods for sending responses
func (n *Nexus) sendPartialResponse(req *Request, content string) {
	resp := &InferenceResponse{
		Type:    ResponseTypePartial,
		Content: content,
		Status:  ResponseStatusSuccess,
	}
	n.sendWrappedResponse(req, resp)
}

func (n *Nexus) sendFinalResponse(req *Request) {
	resp := &InferenceResponse{
		Type:   ResponseTypeFinal,
		Status: ResponseStatusSuccess,
	}
	n.sendWrappedResponse(req, resp)
}

// User response handling
func (n *Nexus) handleSetUser(req *Request) error {
	userReq, ok := req.Data.(*SetUserRequest)
	if !ok {
		return fmt.Errorf("invalid set user request data")
	}

	n.tracker.AddUser(userReq.UserID)

	resp := &SetUserResponse{Status: ResponseStatusSuccess}
	wrapped, err := NewWrappedResponse(req.RequestID, resp)
	if err != nil {
		return fmt.Errorf("failed to wrap response: %w", err)
	}

	n.queue.EnqueueResponse(wrapped, req.ConnectionID)
	return nil
}

func (n *Nexus) sendErrorResponse(req *Request, err error) {
	var resp NexusResponse
	switch req.Type {
	case RequestTypeInference:
		resp = &InferenceResponse{
			Type:    ResponseTypeFinal,
			Content: err.Error(),
			Status:  ResponseStatusError,
		}
	case RequestTypeSetUser:
		resp = &SetUserResponse{
			Status: ResponseStatusError,
		}
	default:
		log.Printf("Cannot send error response for unknown request type: %s", req.Type)
		return
	}

	wrapped, err := NewWrappedResponse(req.RequestID, resp)
	if err != nil {
		log.Printf("Failed to wrap error response: %v", err)
		return
	}

	n.queue.EnqueueResponse(wrapped, req.ConnectionID)
}
