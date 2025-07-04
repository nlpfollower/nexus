package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/nlpfollower/deltamind/database/db"
)

// TrainingJob represents an active training job
type TrainingJob struct {
	ID            string
	JobID         string // From training request
	UserID        db.Digest
	SourceModelID string
	TargetModelID string
	Status        string
	Progress      float64
	Error         string
	StartedAt     time.Time
	CompletedAt   *time.Time
	DatasetPath   string // Path where processed dataset is stored
	NodeCount     int    // Number of nodes used for training
	mu            sync.RWMutex
}

// TrainingDataset represents the parsed training dataset
type TrainingDataset struct {
	ContextMessages []Message `json:"context_messages"`
	TrainingPrompt  string    `json:"training_prompt"`
}

// ProcessDatasetRequest represents the request to process a dataset
type ProcessDatasetRequest struct {
	Dataset     TrainingDataset `json:"dataset"`
	ModelID     string          `json:"model_id"`
	DatasetName string          `json:"dataset_name"`
}

// ProcessDatasetResponse represents the response from dataset processing
type ProcessDatasetResponse struct {
	Success     bool   `json:"success"`
	DatasetPath string `json:"dataset_path"`
	Error       string `json:"error,omitempty"`
}

// TrainingManager manages training jobs
type TrainingManager struct {
	jobs             *sync.Map // map[string]*TrainingJob
	orchestrationMgr *OrchestrationManager
}

// NewTrainingManager creates a new training manager
func NewTrainingManager(orchestrationMgr *OrchestrationManager) *TrainingManager {
	return &TrainingManager{
		jobs:             &sync.Map{},
		orchestrationMgr: orchestrationMgr,
	}
}

// determineNodeCount determines the appropriate node count based on model size
func (tm *TrainingManager) determineNodeCount(modelSize string) int {
	switch strings.ToLower(modelSize) {
	case "70b":
		return 2 // 70B models need 2 nodes
	case "8b", "3b":
		return 1 // Smaller models use 1 node
	default:
		// Default to 1 node for unknown models (safer for resource usage)
		log.Printf("Unknown model size %s, defaulting to 1 node", modelSize)
		return 1
	}
}

// StartTraining processes a training request
func (tm *TrainingManager) StartTraining(ctx context.Context, req *TrainingRequest) (*TrainingJob, error) {
	// Parse the dataset from JSON
	var dataset TrainingDataset
	if err := json.Unmarshal([]byte(req.Dataset), &dataset); err != nil {
		return nil, fmt.Errorf("failed to parse dataset: %w", err)
	}

	// Determine node count based on model size from request
	nodeCount := tm.determineNodeCount(req.ModelSize)

	log.Printf("Starting training job %s for user %s", req.JobID, req.UserID)
	log.Printf("Source model: %s, Target model: %s", req.SourceModelID, req.TargetModelID)
	log.Printf("Node count: %d (for %s model)", nodeCount, req.ModelSize)
	log.Printf("Context messages: %d, Training prompt length: %d",
		len(dataset.ContextMessages), len(dataset.TrainingPrompt))

	// Create training job
	job := &TrainingJob{
		ID:            uuid.New().String(),
		JobID:         req.JobID,
		UserID:        req.UserID,
		SourceModelID: req.SourceModelID,
		TargetModelID: req.TargetModelID,
		Status:        "initializing",
		Progress:      0.0,
		StartedAt:     time.Now(),
		NodeCount:     nodeCount,
	}

	// Store the job
	tm.jobs.Store(req.JobID, job)

	// Start processing in background
	go tm.processTraining(ctx, job, req, dataset)

	return job, nil
}

// GetJobStatus returns the status of a training job
func (tm *TrainingManager) GetJobStatus(jobID string) (*TrainingJob, error) {
	value, ok := tm.jobs.Load(jobID)
	if !ok {
		return nil, fmt.Errorf("training job not found: %s", jobID)
	}

	job := value.(*TrainingJob)
	job.mu.RLock()
	defer job.mu.RUnlock()

	// Return a copy to avoid race conditions
	jobCopy := &TrainingJob{
		ID:            job.ID,
		JobID:         job.JobID,
		UserID:        job.UserID,
		SourceModelID: job.SourceModelID,
		TargetModelID: job.TargetModelID,
		Status:        job.Status,
		Progress:      job.Progress,
		Error:         job.Error,
		StartedAt:     job.StartedAt,
		CompletedAt:   job.CompletedAt,
		DatasetPath:   job.DatasetPath,
		NodeCount:     job.NodeCount,
	}

	return jobCopy, nil
}

// processTraining handles the training workflow
func (tm *TrainingManager) processTraining(ctx context.Context, job *TrainingJob, req *TrainingRequest, dataset TrainingDataset) {
	// Update status to processing
	tm.updateJobStatus(job, "processing_dataset", 0.1, "")

	// Step 1: Get or create an inference session to process the dataset
	log.Printf("Processing dataset for job %s", job.JobID)

	// Use the source model's checkpoint path
	checkpointPath := req.CheckpointPath
	if checkpointPath == "" {
		checkpointPath = fmt.Sprintf("/mnt/cold/contents/dcp/%s/checkpoint", req.SourceModelID)
	}

	// Get an inference session
	session, err := tm.orchestrationMgr.GetOrCreateInferenceSession(ctx, req.SourceModelID, checkpointPath)
	if err != nil {
		tm.updateJobStatus(job, "error", job.Progress, fmt.Sprintf("Failed to get inference session: %v", err))
		return
	}

	// Step 2: Process the dataset using the mindlet server
	datasetName := fmt.Sprintf("dataset-%s", req.TargetModelID)
	datasetPath, err := tm.processDatasetWithMindlet(ctx, session, dataset, req.SourceModelID, datasetName)
	if err != nil {
		tm.updateJobStatus(job, "error", job.Progress, fmt.Sprintf("Failed to process dataset: %v", err))
		return
	}

	job.mu.Lock()
	job.DatasetPath = datasetPath
	job.mu.Unlock()

	tm.updateJobStatus(job, "processed_dataset", 0.3, "")
	log.Printf("Dataset processed and saved to %s for job %s", datasetPath, job.JobID)

	// Step 3: Stop all inference sessions before starting training
	log.Printf("Stopping inference sessions before training for job %s", job.JobID)
	err = tm.stopAllInferenceSessions(ctx)
	if err != nil {
		log.Printf("Warning: Failed to stop some inference sessions: %v", err)
		// Continue anyway, training will handle conflicts
	}

	tm.updateJobStatus(job, "starting_training", 0.4, "")

	// Step 4: Start the actual training process via orchestration
	err = tm.startTrainingProcess(ctx, job, req, datasetPath)
	if err != nil {
		tm.updateJobStatus(job, "error", job.Progress, fmt.Sprintf("Failed to start training: %v", err))
		return
	}

	// Monitor training progress
	tm.monitorTraining(ctx, job)
}

// processDatasetWithMindlet sends the dataset to mindlet for processing
func (tm *TrainingManager) processDatasetWithMindlet(ctx context.Context, session *InferenceSession, dataset TrainingDataset, modelID string, datasetName string) (string, error) {
	// Create the process dataset request
	processReq := ProcessDatasetRequest{
		Dataset:     dataset,
		ModelID:     modelID,
		DatasetName: datasetName,
	}

	// Call the mindlet endpoint to process the dataset
	// This assumes mindlet has a /api/dataset/process endpoint
	reqBody, err := json.Marshal(processReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal dataset request: %w", err)
	}

	// Use the session's HTTP client to make the request
	endpointURL := session.Endpoint.String()
	if !strings.HasSuffix(endpointURL, "/") {
		endpointURL += "/"
	}
	endpointURL += "api/dataset/process"

	req, err := http.NewRequestWithContext(ctx, "POST", endpointURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := session.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send dataset processing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("dataset processing failed: %s - %s", resp.Status, string(body))
	}

	// Parse the response
	var processResp ProcessDatasetResponse
	if err := json.NewDecoder(resp.Body).Decode(&processResp); err != nil {
		return "", fmt.Errorf("failed to parse dataset processing response: %w", err)
	}

	if !processResp.Success {
		return "", fmt.Errorf("dataset processing failed: %s", processResp.Error)
	}

	return processResp.DatasetPath, nil
}

// stopAllInferenceSessions stops all running inference sessions
func (tm *TrainingManager) stopAllInferenceSessions(ctx context.Context) error {
	// Get all running inference jobs from orchestration manager
	runningJobs := tm.orchestrationMgr.GetJobsByType(JobTypeInference)

	var lastError error
	for _, job := range runningJobs {
		job.mu.RLock()
		isRunning := job.Status == JobStatusRunning || job.Status == JobStatusInitializing
		jobID := job.ID
		job.mu.RUnlock()

		if isRunning {
			log.Printf("Stopping inference job %s before training", jobID)
			if err := tm.orchestrationMgr.StopJob(ctx, jobID); err != nil {
				log.Printf("Failed to stop inference job %s: %v", jobID, err)
				lastError = err
			}
		}
	}

	// Wait a bit for jobs to actually stop
	if lastError == nil {
		time.Sleep(5 * time.Second)
	}

	return lastError
}

// startTrainingProcess initiates the actual training via orchestration
func (tm *TrainingManager) startTrainingProcess(ctx context.Context, job *TrainingJob, req *TrainingRequest, datasetPath string) error {
	// Prepare training configuration for orchestration
	trainingConfig := map[string]interface{}{
		"source_model_id": req.SourceModelID,
		"target_model_id": req.TargetModelID,
		"checkpoint_path": req.CheckpointPath,
		"output_path":     req.OutputPath,
		"dataset_path":    datasetPath, // Use the processed dataset path
		"model_name":      req.TargetModelID,
		"model_size":      req.ModelSize, // Pass model size to orchestration
		"node_count":      job.NodeCount, // Use determined node count
	}

	// Start training via orchestration manager
	orchJob, err := tm.orchestrationMgr.StartTrainingJob(ctx, req.TargetModelID, trainingConfig)
	if err != nil {
		return fmt.Errorf("failed to start training orchestration: %w", err)
	}

	// Update job status
	tm.updateJobStatus(job, "training", 0.5, "")

	log.Printf("Training orchestration started for job %s with orchestration ID %s (using %d nodes)",
		job.JobID, orchJob.ID, job.NodeCount)

	return nil
}

// monitorTraining tracks the progress of a training job
func (tm *TrainingManager) monitorTraining(ctx context.Context, job *TrainingJob) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			tm.updateJobStatus(job, "error", job.Progress, "Training cancelled")
			return

		case <-ticker.C:
			// Get training status from orchestration
			status, err := tm.orchestrationMgr.GetTrainingStatus(ctx, job.TargetModelID)
			if err != nil {
				log.Printf("Error checking training status for job %s: %v", job.JobID, err)
				continue
			}

			// Update job based on orchestration status
			if statusStr, ok := status["status"].(string); ok {
				switch statusStr {
				case "completed":
					tm.updateJobStatus(job, "completed", 1.0, "")
					job.mu.Lock()
					now := time.Now()
					job.CompletedAt = &now
					job.mu.Unlock()
					log.Printf("Training job %s completed successfully", job.JobID)
					return

				case "error", "failed":
					errorMsg := "Training failed"
					if errStr, ok := status["error"].(string); ok {
						errorMsg = errStr
					}
					tm.updateJobStatus(job, "error", job.Progress, errorMsg)
					job.mu.Lock()
					now := time.Now()
					job.CompletedAt = &now
					job.mu.Unlock()
					log.Printf("Training job %s failed: %s", job.JobID, errorMsg)
					return

				case "running":
					// Update progress if available
					if progress, ok := status["progress"].(float64); ok {
						// Map orchestration progress to our progress range (0.5 to 1.0)
						mappedProgress := 0.5 + (progress * 0.5)
						tm.updateJobStatus(job, "training", mappedProgress, "")
					}
				}
			}
		}
	}
}

// updateJobStatus updates the status of a training job
func (tm *TrainingManager) updateJobStatus(job *TrainingJob, status string, progress float64, error string) {
	job.mu.Lock()
	defer job.mu.Unlock()

	job.Status = status
	job.Progress = progress
	if error != "" {
		job.Error = error
	}
}

// StopAllJobs stops all active training jobs
func (tm *TrainingManager) StopAllJobs(ctx context.Context) {
	log.Printf("Stopping all training jobs...")

	tm.jobs.Range(func(key, value interface{}) bool {
		job := value.(*TrainingJob)
		job.mu.Lock()
		if job.Status == "training" || job.Status == "processing_dataset" || job.Status == "initializing" {
			job.Status = "stopped"
			job.Error = "Training stopped for shutdown"
			now := time.Now()
			job.CompletedAt = &now
		}
		job.mu.Unlock()
		return true
	})
}
