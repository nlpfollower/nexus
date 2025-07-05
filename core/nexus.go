package core

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"log"
	"os"
	"path/filepath"
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
	Port   int
	LogDir string // Added: Directory for logging requests
}

type Nexus struct {
	queue            *MessageChannel
	tracker          *RequestTracker
	apiManager       *APIModelManager
	orchestrationMgr *OrchestrationManager
	connManager      *ConnectionManager
	trainingMgr      *TrainingManager

	// Add generation tracking
	// key is requestID.String()
	activeGenerations *utils.ConcurrentMap[string, *activeGeneration]

	// Logging configuration
	logDir     string
	runLogDir  string // Directory for this specific run
	logEnabled bool

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
		logDir:            cfg.LogDir,
		logEnabled:        cfg.LogDir != "",
	}

	// Set up logging directory if enabled
	if nexus.logEnabled {
		// Create run-specific directory with timestamp
		runTimestamp := time.Now().Format("2006-01-02_15-04-05")
		nexus.runLogDir = filepath.Join(cfg.LogDir, fmt.Sprintf("run_%s", runTimestamp))

		if err := os.MkdirAll(nexus.runLogDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create log directory %s: %w", nexus.runLogDir, err)
		}

		log.Printf("Request logging enabled. Logs will be saved to: %s", nexus.runLogDir)
	}

	// Create training manager without nexus back reference
	nexus.trainingMgr = NewTrainingManager(orchestrationMgr)

	// Create connection manager with closure callback
	nexus.connManager = NewConnectionManager(
		cfg.Port,
		queue,
		nexus.handleConnectionClosed,
	)

	return nexus, nil
}

// logInferenceRequest saves an inference request to a file
func (n *Nexus) logInferenceRequest(req *Request, inferReq *InferenceRequest) error {
	if !n.logEnabled {
		return nil
	}

	// Create log entry with metadata
	logEntry := struct {
		Timestamp    time.Time `json:"timestamp"`
		RequestID    string    `json:"request_id"`
		ConnectionID string    `json:"connection_id"`
		ModelID      string    `json:"model_id"`
		ModelSize    string    `json:"model_size,omitempty"`
		Checkpoint   string    `json:"checkpoint_path,omitempty"`
		Messages     []Message `json:"messages"`
		RequestType  string    `json:"request_type"`
		IsAPIModel   bool      `json:"is_api_model"`
	}{
		Timestamp:    time.Now(),
		RequestID:    req.RequestID.String(),
		ConnectionID: req.ConnectionID,
		ModelID:      inferReq.ModelID,
		ModelSize:    inferReq.ModelSize,
		Checkpoint:   inferReq.CheckpointPath,
		Messages:     inferReq.Messages,
		RequestType:  "inference",
		IsAPIModel:   n.apiManager.IsAPIModel(db.NewDigest([]byte(inferReq.ModelID))),
	}

	// Generate filename with timestamp and request ID
	timestamp := time.Now().Format("2006-01-02_15-04-05.000")
	filename := fmt.Sprintf("request_%s_%s.json", timestamp, req.RequestID.String()[:8])
	filepath := filepath.Join(n.runLogDir, filename)

	// Marshal to JSON with indentation
	data, err := json.MarshalIndent(logEntry, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write request log: %w", err)
	}

	log.Printf("Logged inference request to: %s", filename)
	return nil
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
		log.Printf("Nexus shutdown initiated...")

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

		// Stop all training jobs
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		n.trainingMgr.StopAllJobs(ctx)
		cancel()

		// CRITICAL: Stop all running jobs before shutdown
		log.Printf("Stopping all running jobs for clean shutdown...")
		runningJobs := n.orchestrationMgr.GetRunningJobs()
		for _, job := range runningJobs {
			job.mu.Lock()
			jobID := job.ID
			status := job.Status
			job.mu.Unlock()

			if status == JobStatusRunning || status == JobStatusInitializing {
				log.Printf("Stopping job %s for clean shutdown", jobID)
				ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
				if err := n.orchestrationMgr.StopJob(ctx, jobID); err != nil {
					log.Printf("Error stopping job %s during shutdown: %v", jobID, err)
				}
				cancel()
			}
		}

		// Stop orchestration manager
		n.orchestrationMgr.Stop()

		// Wait for all goroutines to finish
		n.wg.Wait()

		// Close the request tracker
		n.queue.Close()

		log.Printf("Nexus shutdown completed")
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
	case RequestTypeTraining:
		return n.handleTraining(req)
	case RequestTypeTrainingStatus:
		return n.handleTrainingStatus(req)
	default:
		return fmt.Errorf("unknown request type: %s", req.Type)
	}
}

func (n *Nexus) handleInference(req *Request) error {
	inferReq, ok := req.Data.(*InferenceRequest)
	if !ok {
		return fmt.Errorf("invalid inference request data")
	}

	// Log the inference request
	if err := n.logInferenceRequest(req, inferReq); err != nil {
		// Log error but don't fail the request
		log.Printf("Failed to log inference request: %v", err)
	}

	// Convert string to digest for API model check
	modelDigest := db.NewDigest([]byte(inferReq.ModelID))

	// Check if this is an API model
	if n.apiManager.IsAPIModel(modelDigest) {
		return n.handleAPIInference(req)
	}

	// Use checkpoint path from request if provided, otherwise fall back to model path
	var modelPath string
	if inferReq.CheckpointPath != "" {
		modelPath = inferReq.CheckpointPath
		log.Printf("Using checkpoint path from request: %s", modelPath)
	} else {
		// Fallback for base models that don't have checkpoint path in request
		modelPath = fmt.Sprintf("/mnt/cold/contents/dcp/%s", inferReq.ModelID)
		log.Printf("Using default model path: %s", modelPath)
	}

	// Get or create an inference session for this model
	sessionCtx, sessionCancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer sessionCancel()

	session, err := n.orchestrationMgr.GetOrCreateInferenceSession(sessionCtx, inferReq.ModelID, modelPath)
	if err != nil {
		return fmt.Errorf("failed to get/create inference session: %w", err)
	}

	// Set model size on the session if provided
	if inferReq.ModelSize != "" {
		session.SetModelSize(inferReq.ModelSize)
		log.Printf("Set model size to %s for session %s", inferReq.ModelSize, session.ID)
	}

	// Process the inference request with a fresh context
	inferenceCtx := context.Background()
	stream, err := session.ProcessInference(inferenceCtx, inferReq.Messages)
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
			Port:          9090, // Default mindlet port
			NodeCount:     1,
			RaidMountPath: "/mnt/cold",
			RaidName:      "cold-new",
		}

		job, err := n.orchestrationMgr.StartInferenceJob(ctx, sessionReq.ModelID, config)
		if err != nil {
			return fmt.Errorf("failed to start inference job: %w", err)
		}

		// Return immediately with the job ID and initializing status
		response = SessionResponse{
			Status:    ResponseStatusSuccess,
			SessionID: job.ID,
			State:     string(JobStatusInitializing),
		}

	case SessionActionStop:
		if sessionReq.SessionID == "" {
			return fmt.Errorf("session_id is required for stopping a session")
		}

		// Get the job to check if it exists and update status immediately
		job, ok := n.orchestrationMgr.GetJob(sessionReq.SessionID)
		if !ok {
			return fmt.Errorf("session not found: %s", sessionReq.SessionID)
		}

		job.mu.Lock()
		currentStatus := job.Status
		if currentStatus != JobStatusStopped && currentStatus != JobStatusError && currentStatus != JobStatusStopping {
			job.Status = JobStatusStopping
		}
		job.mu.Unlock()

		// Start the stop operation in a goroutine with proper completion tracking
		go func() {
			stopCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			log.Printf("Starting stop operation for job %s", sessionReq.SessionID)
			if err := n.orchestrationMgr.StopJob(stopCtx, sessionReq.SessionID); err != nil {
				log.Printf("Error stopping job %s: %v", sessionReq.SessionID, err)
				// Update status on error
				job.mu.Lock()
				job.Status = JobStatusError
				job.LastError = err
				now := time.Now()
				job.StoppedAt = &now
				job.mu.Unlock()
			} else {
				// Successfully stopped - update to final stopped state
				job.mu.Lock()
				job.Status = JobStatusStopped
				now := time.Now()
				job.StoppedAt = &now
				job.mu.Unlock()
				log.Printf("Successfully completed stop operation for job %s", sessionReq.SessionID)
			}
		}()

		// Return immediately with stopping status
		response = SessionResponse{
			Status:    ResponseStatusSuccess,
			SessionID: sessionReq.SessionID,
			State:     string(JobStatusStopping),
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
		status := job.Status
		job.mu.RUnlock()

		response = SessionResponse{
			Status:    ResponseStatusSuccess,
			SessionID: job.ID,
			Endpoint:  endpoint,
			State:     string(status),
		}

	case SessionActionStatus:
		// Check session status
		if sessionReq.SessionID == "" {
			return fmt.Errorf("session_id is required for checking status")
		}

		job, ok := n.orchestrationMgr.GetJob(sessionReq.SessionID)
		if !ok {
			return fmt.Errorf("session not found: %s", sessionReq.SessionID)
		}

		job.mu.RLock()
		endpoint := job.Endpoint
		status := job.Status
		job.mu.RUnlock()

		response = SessionResponse{
			Status:    ResponseStatusSuccess,
			SessionID: job.ID,
			Endpoint:  endpoint,
			State:     string(status),
		}

	default:
		return fmt.Errorf("unknown session action: %s", sessionReq.Action)
	}

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

	modelDigest := db.NewDigest([]byte(inferReq.ModelID))
	model, err := n.apiManager.GetAPIModel(modelDigest)
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

func (n *Nexus) handleTraining(req *Request) error {
	trainReq, ok := req.Data.(*TrainingRequest)
	if !ok {
		return fmt.Errorf("invalid training request data")
	}

	// Start the training job via training manager
	trainingJob, err := n.trainingMgr.StartTraining(context.Background(), trainReq)
	if err != nil {
		return fmt.Errorf("failed to start training: %w", err)
	}

	// Send initial response
	resp := &TrainingResponse{
		JobID:  trainReq.JobID,
		Status: trainingJob.Status,
	}
	n.sendWrappedResponse(req, resp)
	return nil
}

// handleTrainingStatus handles training status requests
func (n *Nexus) handleTrainingStatus(req *Request) error {
	statusReq, ok := req.Data.(*TrainingStatusRequest)
	if !ok {
		return fmt.Errorf("invalid training status request data")
	}

	// Get job status from training manager
	// The training manager will check orchestration for completed jobs
	job, err := n.trainingMgr.GetJobStatus(statusReq.JobID)
	if err != nil {
		return fmt.Errorf("failed to get training status: %w", err)
	}

	// Convert to training status response
	resp := &TrainingStatusResponse{
		JobID:       job.JobID,
		Status:      job.Status,
		Progress:    job.Progress,
		Error:       job.Error,
		StartedAt:   job.StartedAt,
		CompletedAt: job.CompletedAt,
	}

	n.sendWrappedResponse(req, resp)
	return nil
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
	case RequestTypeTraining:
		resp = &TrainingResponse{
			JobID:  "",
			Status: "error",
		}
	case RequestTypeTrainingStatus:
		resp = &TrainingStatusResponse{
			JobID:  "",
			Status: "error",
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
