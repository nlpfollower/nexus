// core/nexus.go
package core

import (
	"context"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"log"
	"strings"
	"sync"
	"time"
)

type activeGeneration struct {
	stream       GenerationStream
	requestID    db.Digest
	connectionID string
}

type Config struct {
	Port int
}

type Nexus struct {
	queue            *MessageChannel
	tracker          *RequestTracker
	apiManager       *APIModelManager
	orchestrationMgr *OrchestrationManager
	connManager      *ConnectionManager

	// Add generation tracking
	// key is requestID.String()
	activeGenerations *utils.ConcurrentMap[string, *activeGeneration]

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

	// Create orchestration manager with default 30 minute expiration
	orchestrationMgr, err := NewOrchestrationManager(30 * time.Minute)
	if err != nil {
		return nil, fmt.Errorf("failed to create orchestration manager: %w", err)
	}

	nexus := &Nexus{
		queue:             queue,
		tracker:           NewRequestTracker(),
		apiManager:        apiManager,
		orchestrationMgr:  orchestrationMgr,
		activeGenerations: utils.NewConcurrentMap[string, *activeGeneration](),
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

	// Start orchestration manager
	n.orchestrationMgr.Start()

	// Start request processing loop
	n.wg.Add(1)
	go func() {
		defer n.wg.Done()
		n.requestLoop()
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

		// Stop orchestration manager
		n.orchestrationMgr.Stop()

		// Wait for all goroutines to finish
		n.wg.Wait()

		// Close the request tracker
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
// requestLoop processes incoming requests
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
			// Process the request directly
			go func(req *Request) {
				if err := n.processRequest(req); err != nil {
					log.Printf("Error processing request: %v", err)
					n.sendErrorResponse(req, err)
				}
			}(req)
		}
	}
}

// processRequest handles a request based on its type
func (n *Nexus) processRequest(req *Request) error {
	n.tracker.AddRequest(req)

	switch req.Type {
	case RequestTypeInference:
		return n.handleInference(req)
	case RequestTypeSetUser:
		return n.handleSetUser(req)
	case RequestTypeSession:
		return n.handleSessionRequest(req)
	case RequestTypeJobStatus:
		return n.handleJobStatus(req)
	default:
		return fmt.Errorf("unknown request type: %s", req.Type)
	}
}

// handleInference processes an inference request
func (n *Nexus) handleInference(req *Request) error {
	inferReq, ok := req.Data.(*InferenceRequest)
	if !ok {
		return fmt.Errorf("invalid inference request data")
	}

	// Check if this is an API model
	if n.apiManager.IsAPIModel(inferReq.ModelID) {
		return n.handleAPIInference(req)
	}

	// Get or create an inference session for this model
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	session, err := n.orchestrationMgr.GetOrCreateInferenceSession(ctx, inferReq.ModelID.String())
	if err != nil {
		return fmt.Errorf("failed to get/create inference session: %w", err)
	}

	// Process the inference request
	stream, err := session.ProcessInference(ctx, inferReq.Messages)
	if err != nil {
		return fmt.Errorf("failed to process inference: %w", err)
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

func (n *Nexus) handleSessionRequest(req *Request) error {
	sessionReq, ok := req.Data.(*SessionRequest)
	if !ok {
		return fmt.Errorf("invalid session request data")
	}

	ctx := context.Background()
	var response SessionResponse

	switch sessionReq.Action {
	case SessionActionStart:
		// Start a new inference job
		config := InferenceConfig{
			DCPDir:           fmt.Sprintf("/mnt/cold/contents/dcp/%s/step-0", sessionReq.ModelID),
			TokenizerPath:    "/mnt/cold/contents/checkpoints/Llama3.1-8B-Instruct/tokenizer.model",
			ParamsPath:       "torchchat/model_params/Meta-Llama-3.1-8B.json",
			Port:             5000,
			DCPModelSize:     "8B",
			CheckpointFolder: sessionReq.ModelID,
			NodeCount:        1,
			RaidMountPath:    "/mnt/cold",
			RaidName:         "cold-new",
		}

		job, err := n.orchestrationMgr.StartInferenceJob(ctx, sessionReq.ModelID, config)
		if err != nil {
			return fmt.Errorf("failed to start inference job: %w", err)
		}

		// Wait for job to be ready
		readyCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
		defer cancel()

		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-readyCtx.Done():
				return fmt.Errorf("timeout waiting for inference job to start")
			case <-ticker.C:
				job.mu.RLock()
				status := job.Status
				endpoint := job.Endpoint
				job.mu.RUnlock()

				if status == JobStatusRunning && endpoint != "" {
					response = SessionResponse{
						Status:    ResponseStatusSuccess,
						SessionID: job.ID,
						Endpoint:  endpoint,
					}
					goto sendResponse
				} else if status == JobStatusError {
					return fmt.Errorf("inference job failed to start")
				}
			}
		}

	case SessionActionStop:
		if sessionReq.SessionID == "" {
			return fmt.Errorf("session_id is required for stopping a session")
		}

		// Stop the specified job
		if err := n.orchestrationMgr.StopJob(ctx, sessionReq.SessionID); err != nil {
			return fmt.Errorf("failed to stop job: %w", err)
		}

		response = SessionResponse{
			Status:    ResponseStatusSuccess,
			SessionID: sessionReq.SessionID,
		}

	case SessionActionExtend:
		if sessionReq.SessionID == "" {
			return fmt.Errorf("session_id is required for extending a session")
		}

		// Parse duration if provided
		var duration time.Duration
		if sessionReq.Duration != "" {
			var err error
			duration, err = time.ParseDuration(sessionReq.Duration)
			if err != nil {
				return fmt.Errorf("invalid duration format: %w", err)
			}
		} else {
			duration = 30 * time.Minute // default extension
		}

		// Extend the specified job
		if err := n.orchestrationMgr.ExtendJob(sessionReq.SessionID, duration); err != nil {
			return fmt.Errorf("failed to extend job: %w", err)
		}

		// Get the job to return its details
		job, ok := n.orchestrationMgr.GetJob(sessionReq.SessionID)
		if !ok {
			return fmt.Errorf("job not found after extension: %s", sessionReq.SessionID)
		}

		job.mu.RLock()
		endpoint := job.Endpoint
		job.mu.RUnlock()

		response = SessionResponse{
			Status:    ResponseStatusSuccess,
			SessionID: job.ID,
			Endpoint:  endpoint,
		}

	default:
		return fmt.Errorf("unknown session action: %s", sessionReq.Action)
	}

sendResponse:
	// Send the response
	n.sendWrappedResponse(req, &response)
	return nil
}

func (n *Nexus) handleJobStatus(req *Request) error {
	jobReq, ok := req.Data.(*JobStatusRequest)
	if !ok {
		return fmt.Errorf("invalid job status request data")
	}

	info, err := n.orchestrationMgr.GetJobStatus(context.Background(), jobReq.JobID)
	if err != nil {
		resp := &JobStatusResponse{
			Status: ResponseStatusError,
			Error:  err.Error(),
		}
		n.sendWrappedResponse(req, resp)
		return nil
	}

	resp := &JobStatusResponse{
		Status:  ResponseStatusSuccess,
		JobInfo: info,
	}
	n.sendWrappedResponse(req, resp)
	return nil
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

	log.Printf("Starting to process stream for request %s", key)

	responseChan := stream.ResponseChan()
	for {
		select {
		case <-n.done:
			log.Printf("Nexus shutting down, stopping generation %s", key)
			return

		case resp, ok := <-responseChan:
			if !ok {
				log.Printf("Response channel closed for %s, sending final response", key)
				n.sendFinalResponse(req)
				return
			}

			if resp.Error != nil {
				// Don't treat context canceled as a real error
				if strings.Contains(resp.Error.Error(), "context canceled") {
					log.Printf("Context canceled for %s, sending final response", key)
					n.sendFinalResponse(req)
					return
				}

				log.Printf("Error from generation %s: %v", key, resp.Error)
				n.sendErrorResponse(req, resp.Error)
				return
			}

			if resp.Content != "" {
				log.Printf("Received content chunk for %s: %q", key, resp.Content)
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

func (n *Nexus) sendErrorResponse(req *Request, err error) {
	var resp NexusResponse

	// Don't treat context cancellation as a real error
	errorContent := err.Error()
	if strings.Contains(errorContent, "context canceled") {
		errorContent = ""
	}

	switch req.Type {
	case RequestTypeInference:
		resp = &InferenceResponse{
			Type:    ResponseTypeFinal,
			Content: errorContent,
			Status:  ResponseStatusError,
		}
	case RequestTypeSetUser:
		resp = &SetUserResponse{
			Status: ResponseStatusError,
		}
	case RequestTypeSession:
		resp = &SessionResponse{
			Status: ResponseStatusError,
			Error:  errorContent,
		}
	case RequestTypeJobStatus:
		resp = &JobStatusResponse{
			Status: ResponseStatusError,
			Error:  errorContent,
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
