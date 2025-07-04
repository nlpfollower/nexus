// core/orchestration_manager.go
package core

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/nlpfollower/deltamind/orchestration/utils"
)

// OrchestrationJob represents a managed inference or training job
type OrchestrationJob struct {
	ID         string
	Type       OrchestrationJobType
	Status     OrchestrationJobStatus
	ModelID    string
	Config     map[string]interface{}
	CreatedAt  time.Time
	StartedAt  *time.Time
	StoppedAt  *time.Time
	Expiration time.Time
	LastError  error
	Endpoint   string // For inference jobs
	mu         sync.RWMutex
}

// InferenceConfig contains inference-specific configuration
type InferenceConfig struct {
	DCPDir           string `json:"dcp_dir"`
	TokenizerPath    string `json:"tokenizer_path"`
	ParamsPath       string `json:"params_path"`
	Port             int    `json:"port"`
	DCPModelSize     string `json:"dcp_model_size"`
	CheckpointFolder string `json:"checkpoint_folder"`
	NodeCount        int    `json:"node_count"`
	RaidMountPath    string `json:"raid_mount_path"`
	RaidName         string `json:"raid_name"`
}

// OrchestrationManager manages inference and training jobs via the orchestration CLI
type OrchestrationManager struct {
	jobs              *utils.ConcurrentMap[string, *OrchestrationJob]
	defaultExpiration time.Duration
	orchestrationDir  string // Directory where orchestration code lives

	done      chan struct{}
	wg        sync.WaitGroup
	closeOnce sync.Once
}

// NewOrchestrationManager creates a new orchestration manager
func NewOrchestrationManager(defaultExpiration time.Duration) (*OrchestrationManager, error) {
	// Find orchestration directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get home directory: %w", err)
	}

	orchestrationDir := filepath.Join(homeDir, "orchestration")
	if _, err := os.Stat(orchestrationDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("orchestration directory not found at %s", orchestrationDir)
	}

	manager := &OrchestrationManager{
		jobs:              utils.NewConcurrentMap[string, *OrchestrationJob](),
		defaultExpiration: defaultExpiration,
		orchestrationDir:  orchestrationDir,
		done:              make(chan struct{}),
	}

	return manager, nil
}

// Start begins the orchestration manager's background processes
func (m *OrchestrationManager) Start() {
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		m.expirationMonitor()
	}()

	log.Println("Orchestration manager started")
}

// Stop gracefully shuts down the orchestration manager
func (m *OrchestrationManager) Stop() {
	m.closeOnce.Do(func() {
		log.Println("Orchestration manager stopping...")
		close(m.done)

		// Stop all running jobs with proper cleanup
		runningJobs := m.GetRunningJobs()
		if len(runningJobs) > 0 {
			log.Printf("Stopping %d running jobs for clean shutdown", len(runningJobs))

			for _, job := range runningJobs {
				job.mu.RLock()
				jobID := job.ID
				jobType := job.Type
				status := job.Status
				job.mu.RUnlock()

				if status == JobStatusRunning || status == JobStatusInitializing {
					log.Printf("Stopping %s job %s", jobType, jobID)
					ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
					if err := m.stopJob(ctx, job); err != nil {
						log.Printf("Error stopping job %s: %v", jobID, err)
					} else {
						log.Printf("Successfully stopped job %s", jobID)
					}
					cancel()
				}
			}
		}

		m.wg.Wait()
		log.Println("Orchestration manager stopped")
	})
}

// StartInferenceJob starts a new inference job
func (m *OrchestrationManager) StartInferenceJob(ctx context.Context, modelID string, config InferenceConfig) (*OrchestrationJob, error) {
	jobID := uuid.New().String()

	// Convert config to map
	configMap := make(map[string]interface{})
	configBytes, _ := json.Marshal(config)
	json.Unmarshal(configBytes, &configMap)

	job := &OrchestrationJob{
		ID:         jobID,
		Type:       JobTypeInference,
		Status:     JobStatusPending,
		ModelID:    modelID,
		Config:     configMap,
		CreatedAt:  time.Now(),
		Expiration: time.Now().Add(m.defaultExpiration),
	}

	m.jobs.Set(jobID, job)

	// Start the job asynchronously
	go func() {
		if err := m.startInferenceProcess(ctx, job, config); err != nil {
			job.mu.Lock()
			job.Status = JobStatusError
			job.LastError = err
			job.mu.Unlock()
			log.Printf("Failed to start inference job %s: %v", jobID, err)
		}
	}()

	return job, nil
}

func (m *OrchestrationManager) StartTrainingJob(ctx context.Context, targetModelID string, config map[string]interface{}) (*OrchestrationJob, error) {
	jobID := uuid.New().String()

	job := &OrchestrationJob{
		ID:         jobID,
		Type:       JobTypeTraining,
		Status:     JobStatusPending,
		ModelID:    targetModelID,
		Config:     config,
		CreatedAt:  time.Now(),
		Expiration: time.Now().Add(m.defaultExpiration * 4), // Training jobs need more time
	}

	m.jobs.Set(jobID, job)

	// Start the job asynchronously
	go func() {
		if err := m.startTrainingProcess(ctx, job); err != nil {
			job.mu.Lock()
			job.Status = JobStatusError
			job.LastError = err
			job.mu.Unlock()
			log.Printf("Failed to start training job %s: %v", jobID, err)
		}
	}()

	return job, nil
}

// GetJob retrieves a job by ID
func (m *OrchestrationManager) GetJob(jobID string) (*OrchestrationJob, bool) {
	return m.jobs.Get(jobID)
}

// GetJobsByType returns all jobs of a specific type
func (m *OrchestrationManager) GetJobsByType(jobType OrchestrationJobType) []*OrchestrationJob {
	var jobs []*OrchestrationJob
	for _, job := range m.jobs.GetAll() {
		if job.Type == jobType {
			jobs = append(jobs, job)
		}
	}
	return jobs
}

// GetRunningJobs returns all currently running jobs
func (m *OrchestrationManager) GetRunningJobs() []*OrchestrationJob {
	var jobs []*OrchestrationJob
	for _, job := range m.jobs.GetAll() {
		job.mu.RLock()
		isRunning := job.Status == JobStatusRunning || job.Status == JobStatusInitializing
		job.mu.RUnlock()

		if isRunning {
			jobs = append(jobs, job)
		}
	}
	return jobs
}

// StopJob stops a specific job
func (m *OrchestrationManager) StopJob(ctx context.Context, jobID string) error {
	job, exists := m.jobs.Get(jobID)
	if !exists {
		return fmt.Errorf("job not found: %s", jobID)
	}

	return m.stopJob(ctx, job)
}

func (m *OrchestrationManager) stopJob(ctx context.Context, job *OrchestrationJob) error {
	job.mu.Lock()
	currentStatus := job.Status
	jobType := job.Type
	jobID := job.ID
	job.mu.Unlock()

	// Don't stop if already stopped or stopping
	if currentStatus == JobStatusStopped {
		log.Printf("Job %s already stopped", jobID)
		return nil
	}

	if currentStatus == JobStatusError {
		log.Printf("Job %s already in error state", jobID)
		return nil
	}

	log.Printf("Stopping %s job %s (current status: %s)", jobType, jobID, currentStatus)

	var err error
	switch jobType {
	case JobTypeInference:
		err = m.stopInferenceProcess(ctx, job)
	case JobTypeTraining:
		err = m.stopTrainingProcess(ctx, job)
	}

	// Update final status
	job.mu.Lock()
	now := time.Now()
	job.StoppedAt = &now
	if err != nil {
		job.Status = JobStatusError
		job.LastError = err
		log.Printf("Job %s stopped with error: %v", jobID, err)
	} else {
		job.Status = JobStatusStopped
		log.Printf("Job %s stopped successfully", jobID)
	}
	job.mu.Unlock()

	return err
}

func (m *OrchestrationManager) stopInferenceProcess(ctx context.Context, job *OrchestrationJob) error {
	log.Printf("Stopping inference job %s by scaling down", job.ID)

	// Get config from job
	var config InferenceConfig
	configBytes, _ := json.Marshal(job.Config)
	json.Unmarshal(configBytes, &config)

	// Scale down to stop the server
	args := []string{
		"mindlet", "inference",
		"--skip-cluster-creation",
		"--scale-down",
	}

	// Add necessary config parameters for scale-down
	if config.RaidMountPath != "" {
		args = append(args, "--raid-mount-path", config.RaidMountPath)
	}
	if config.RaidName != "" {
		args = append(args, "--raid-name", config.RaidName)
	}

	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		return fmt.Errorf("failed to scale down inference: %w\nOutput: %s", err, string(output))
	}

	log.Printf("Inference job %s stopped successfully", job.ID)
	return nil
}

func (m *OrchestrationManager) stopTrainingProcess(ctx context.Context, job *OrchestrationJob) error {
	log.Printf("Stopping training job %s", job.ID)

	args := []string{
		"mindlet", "train", "stop",
		"--skip-cluster-creation",
	}

	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		// Log but don't fail if stop command has issues
		log.Printf("Training stop command returned error: %v\nOutput: %s", err, string(output))
		// Still mark as successful stop since the command executed
		return nil
	}

	log.Printf("Training job %s stopped successfully", job.ID)
	return nil
}

// ExtendJob extends the expiration time of a job
func (m *OrchestrationManager) ExtendJob(jobID string, duration time.Duration) error {
	job, exists := m.jobs.Get(jobID)
	if !exists {
		return fmt.Errorf("job not found: %s", jobID)
	}

	job.mu.Lock()
	defer job.mu.Unlock()

	if job.Status != JobStatusRunning {
		return fmt.Errorf("can only extend running jobs")
	}

	job.Expiration = time.Now().Add(duration)
	log.Printf("Extended job %s expiration to %v", jobID, job.Expiration)

	return nil
}

// GetJobStatus retrieves the current status of a job
func (m *OrchestrationManager) GetJobStatus(ctx context.Context, jobID string) (*JobStatusInfo, error) {
	job, exists := m.jobs.Get(jobID)
	if !exists {
		return nil, fmt.Errorf("job not found: %s", jobID)
	}

	job.mu.RLock()
	info := &JobStatusInfo{
		ID:         job.ID,
		Type:       job.Type,
		Status:     job.Status,
		ModelID:    job.ModelID,
		CreatedAt:  job.CreatedAt,
		StartedAt:  job.StartedAt,
		StoppedAt:  job.StoppedAt,
		Expiration: job.Expiration,
		Endpoint:   job.Endpoint,
	}
	jobType := job.Type
	job.mu.RUnlock()

	// For inference jobs, check live status
	if jobType == JobTypeInference && info.Status == JobStatusRunning {
		liveStatus, err := m.checkInferenceStatus(ctx)
		if err != nil {
			info.LiveStatus = map[string]interface{}{
				"error": err.Error(),
			}
		} else {
			info.LiveStatus = liveStatus
		}
	}

	return info, nil
}

// GetOrCreateInferenceSession gets or creates an inference session for the given model
func (m *OrchestrationManager) GetOrCreateInferenceSession(ctx context.Context, modelID string, modelPath string) (*InferenceSession, error) {
	// Check for existing running inference jobs (mindlet servers)
	for _, job := range m.jobs.GetAll() {
		job.mu.RLock()
		isRunning := job.Status == JobStatusRunning && job.Type == JobTypeInference
		endpoint := job.Endpoint
		jobID := job.ID
		expiration := job.Expiration
		job.mu.RUnlock()

		if isRunning && endpoint != "" {
			// Parse the endpoint URL
			endpointURL, err := url.Parse(endpoint)
			if err != nil {
				continue
			}

			// The mindlet server can handle multiple models, so we can reuse it
			session := NewInferenceSession(jobID, modelID, time.Until(expiration), false)
			session.SetCheckpointPath(modelPath)
			session.SetModelSize("8B") // Default size, could be made configurable
			session.Status = SessionStatusRunning
			session.Endpoint = endpointURL
			session.Expiration = expiration

			log.Printf("Reusing existing mindlet session %s for model %s at %s", jobID, modelID, endpoint)
			if modelPath != "" && modelPath != modelID {
				log.Printf("Using checkpoint path: %s", modelPath)
			}
			return session, nil
		}
	}

	// No existing session, start a new mindlet server
	// Simplified config - mindlet handles model details internally
	config := InferenceConfig{
		Port:          9090, // Default mindlet port
		NodeCount:     1,
		RaidMountPath: "/mnt/cold",
		RaidName:      "cold-new",
	}

	job, err := m.StartInferenceJob(ctx, modelID, config)
	if err != nil {
		return nil, fmt.Errorf("failed to start mindlet server: %w", err)
	}

	// Wait for the job to be ready
	readyCtx, cancel := context.WithTimeout(ctx, 6*time.Minute)
	defer cancel()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	checkCount := 0
	for {
		select {
		case <-readyCtx.Done():
			return nil, fmt.Errorf("timeout waiting for mindlet server to be ready after %d checks", checkCount)
		case <-ticker.C:
			checkCount++
			job.mu.RLock()
			status := job.Status
			endpoint := job.Endpoint
			lastError := job.LastError
			job.mu.RUnlock()

			log.Printf("GetOrCreateSession check %d: status=%s, endpoint=%s", checkCount, status, endpoint)

			if status == JobStatusError {
				return nil, fmt.Errorf("mindlet server failed to start: %w", lastError)
			}

			if status == JobStatusRunning && endpoint != "" {
				endpointURL, err := url.Parse(endpoint)
				if err != nil {
					return nil, fmt.Errorf("invalid endpoint URL: %w", err)
				}

				session := NewInferenceSession(job.ID, modelID, time.Until(job.Expiration), false)
				session.SetCheckpointPath(modelPath)
				session.SetModelSize("8B") // Default size, could be made configurable
				session.Status = SessionStatusRunning
				session.Endpoint = endpointURL
				session.Expiration = job.Expiration

				log.Printf("Created new mindlet session %s for model %s at %s after %d checks", job.ID, modelID, endpoint, checkCount)
				if modelPath != "" && modelPath != modelID {
					log.Printf("Using checkpoint path: %s", modelPath)
				}
				return session, nil
			}
		}
	}
}

// GetTrainingStatus gets the status of a training job from orchestration
func (m *OrchestrationManager) GetTrainingStatus(ctx context.Context, jobID string) (map[string]interface{}, error) {
	job, exists := m.jobs.Get(jobID)
	if !exists {
		return nil, fmt.Errorf("training job not found: %s", jobID)
	}

	job.mu.RLock()
	status := job.Status
	job.mu.RUnlock()

	// For running training jobs, check live status
	if status == JobStatusRunning {
		liveStatus, err := m.checkTrainingStatus(ctx, jobID)
		if err != nil {
			// Return basic status if live check fails
			return map[string]interface{}{
				"status": string(status),
				"job_id": jobID,
			}, nil
		}

		// Add progress calculation based on training status
		if trainingStatus, ok := liveStatus["status"].(string); ok {
			result := map[string]interface{}{
				"status": trainingStatus,
				"job_id": jobID,
			}

			// Copy any additional fields from live status
			for k, v := range liveStatus {
				if k != "status" && k != "job_id" {
					result[k] = v
				}
			}

			return result, nil
		}

		return liveStatus, nil
	}

	// Return basic status for non-running jobs
	result := map[string]interface{}{
		"status": string(status),
		"job_id": jobID,
	}

	// Add error information if available
	if status == JobStatusError {
		job.mu.RLock()
		if job.LastError != nil {
			result["error"] = job.LastError.Error()
		}
		job.mu.RUnlock()
	}

	return result, nil
}

// executeOrchestrationCommand runs a command in the orchestration directory
func (m *OrchestrationManager) executeOrchestrationCommand(ctx context.Context, args ...string) ([]byte, error) {
	fullArgs := append([]string{"run", "main.go"}, args...)
	cmd := exec.CommandContext(ctx, "go", fullArgs...)
	cmd.Dir = m.orchestrationDir

	// Log the full command
	log.Printf("Executing command in %s: go %s", m.orchestrationDir, strings.Join(fullArgs, " "))

	// Create pipes for both stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start command: %w", err)
	}

	// Capture output while streaming
	var outputBuffer bytes.Buffer
	var errorBuffer bytes.Buffer

	// Create a wait group for the goroutines
	var wg sync.WaitGroup
	wg.Add(2)

	// Stream stdout
	go func() {
		defer wg.Done()
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			log.Printf("[orchestration stdout] %s", line)
			outputBuffer.WriteString(line + "\n")
		}
	}()

	// Stream stderr
	go func() {
		defer wg.Done()
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			log.Printf("[orchestration stderr] %s", line)
			errorBuffer.WriteString(line + "\n")
		}
	}()

	// Wait for command to complete
	cmdErr := cmd.Wait()

	// Wait for all output to be captured
	wg.Wait()

	// Combine stdout and stderr
	combinedOutput := outputBuffer.Bytes()
	if errorBuffer.Len() > 0 {
		combinedOutput = append(combinedOutput, errorBuffer.Bytes()...)
	}

	if cmdErr != nil {
		log.Printf("Command failed with error: %v", cmdErr)
		return combinedOutput, cmdErr
	}

	return combinedOutput, nil
}

func (m *OrchestrationManager) startInferenceProcess(ctx context.Context, job *OrchestrationJob, config InferenceConfig) error {
	job.mu.Lock()
	job.Status = JobStatusInitializing
	now := time.Now()
	job.StartedAt = &now
	job.mu.Unlock()

	// Build the command arguments for mindlet inference
	// Note: We're simplifying the config since mindlet handles model details internally
	args := []string{
		"mindlet", "inference",
		"--node-count", fmt.Sprintf("%d", config.NodeCount),
		"--skip-cluster-creation",
		"--port", fmt.Sprintf("%d", config.Port),
	}

	// Only add mount paths if specified
	if config.RaidMountPath != "" {
		args = append(args, "--raid-mount-path", config.RaidMountPath)
	}
	if config.RaidName != "" {
		args = append(args, "--raid-name", config.RaidName)
	}

	// Log the simplified config being used
	log.Printf("Starting mindlet inference server:")
	log.Printf("  Port: %d", config.Port)
	log.Printf("  NodeCount: %d", config.NodeCount)
	if config.RaidMountPath != "" {
		log.Printf("  RaidMountPath: %s", config.RaidMountPath)
	}
	if config.RaidName != "" {
		log.Printf("  RaidName: %s", config.RaidName)
	}

	// Execute the command
	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		job.mu.Lock()
		job.Status = JobStatusError
		job.LastError = fmt.Errorf("failed to start mindlet server: %w\nOutput: %s", err, string(output))
		job.mu.Unlock()
		return job.LastError
	}

	// Wait for server to be ready with up to 6 minute timeout
	readyCtx, cancel := context.WithTimeout(ctx, 6*time.Minute)
	defer cancel()

	log.Printf("Waiting for mindlet server to be ready (job %s)...", job.ID)

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	checkCount := 0
	for {
		select {
		case <-readyCtx.Done():
			job.mu.Lock()
			job.Status = JobStatusError
			job.LastError = fmt.Errorf("timeout waiting for mindlet server to be ready after %d checks", checkCount)
			job.mu.Unlock()
			return job.LastError

		case <-ticker.C:
			checkCount++
			status, err := m.checkInferenceStatus(ctx)
			if err != nil {
				log.Printf("Check %d: Error checking mindlet status: %v", checkCount, err)
			} else {
				log.Printf("Check %d: Mindlet status: %+v", checkCount, status)
				if serverStatus, ok := status["status"].(string); ok && serverStatus == "healthy" {
					// Server is ready
					job.mu.Lock()
					job.Status = JobStatusRunning
					if endpoint, ok := status["endpoint"].(string); ok {
						job.Endpoint = endpoint
					}
					job.mu.Unlock()

					log.Printf("Mindlet server for job %s started successfully after %d checks", job.ID, checkCount)
					return nil
				}
			}
			// Continue waiting
		}
	}
}

func (m *OrchestrationManager) checkInferenceStatus(ctx context.Context) (map[string]interface{}, error) {
	args := []string{
		"mindlet", "inference", "status",
		"--skip-cluster-creation", "--silent-mode",
	}

	log.Printf("Checking inference status...")
	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		// Status command might return non-zero exit code even with valid JSON output
		// Try to parse the output anyway
		var status map[string]interface{}
		if jsonErr := json.Unmarshal(output, &status); jsonErr == nil {
			log.Printf("Status check returned (with error): %+v", status)
			return status, nil
		}
		log.Printf("Status check failed: %v, output: %s", err, string(output))
		return nil, fmt.Errorf("failed to get inference status: %w", err)
	}

	var status map[string]interface{}
	if err := json.Unmarshal(output, &status); err != nil {
		log.Printf("Failed to parse status JSON: %v, output: %s", err, string(output))
		return nil, fmt.Errorf("failed to parse status output: %w", err)
	}

	log.Printf("Status check returned: %+v", status)
	return status, nil
}

func (m *OrchestrationManager) expirationMonitor() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.done:
			return
		case <-ticker.C:
			m.checkExpirations()
		}
	}
}

func (m *OrchestrationManager) checkExpirations() {
	now := time.Now()

	for _, job := range m.jobs.GetAll() {
		job.mu.RLock()
		shouldExpire := (job.Status == JobStatusRunning || job.Status == JobStatusInitializing) &&
			now.After(job.Expiration)
		jobID := job.ID
		job.mu.RUnlock()

		if shouldExpire {
			log.Printf("Job %s has expired, scaling down to save resources...", jobID)

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			if err := m.stopJob(ctx, job); err != nil {
				log.Printf("Error stopping expired job %s: %v", jobID, err)
			}

			job.mu.Lock()
			job.Status = JobStatusExpired
			job.mu.Unlock()

			cancel()
		}
	}
}

func (m *OrchestrationManager) getNodeCountFromConfig(config map[string]interface{}) int {
	// First check if node_count is explicitly set in config
	if nodeCount, ok := config["node_count"].(int); ok {
		return nodeCount
	}

	// Use model size from config
	if modelSize, ok := config["model_size"].(string); ok {
		return m.determineNodeCountFromModelSize(modelSize)
	}

	// Default to 1 node if nothing else is specified
	return 1
}

// determineNodeCountFromModelSize determines appropriate node count based on model size
func (m *OrchestrationManager) determineNodeCountFromModelSize(modelSize string) int {
	switch strings.ToLower(modelSize) {
	case "70b":
		return 2 // 70B models need 2 nodes
	case "8b", "3b":
		return 1 // Smaller models use 1 node
	default:
		// Default to 1 node for unknown models
		log.Printf("Unknown model size %s, defaulting to 1 node", modelSize)
		return 1
	}
}

// Updated startTrainingProcess to use dynamic node count
func (m *OrchestrationManager) startTrainingProcess(ctx context.Context, job *OrchestrationJob) error {
	job.mu.Lock()
	job.Status = JobStatusInitializing
	now := time.Now()
	job.StartedAt = &now
	job.mu.Unlock()

	// Extract config
	sourceModelID := job.Config["source_model_id"].(string)
	targetModelID := job.Config["target_model_id"].(string)
	checkpointPath := job.Config["checkpoint_path"].(string)
	outputPath := job.Config["output_path"].(string)
	datasetPath := job.Config["dataset_path"].(string)
	modelName := job.Config["model_name"].(string)
	modelSize := job.Config["model_size"].(string)
	modelSize = strings.ToLower(modelSize)

	// Determine node count
	nodeCount := m.getNodeCountFromConfig(job.Config)

	// Build command arguments for mindlet train
	args := []string{
		"mindlet", "train",
		"--model-path", checkpointPath,
		"--dataset-path", datasetPath,
		"--output-dir", outputPath,
		"--model-name", modelName,
		"--model-size", modelSize,
		"--node-count", fmt.Sprintf("%d", nodeCount),
		"--skip-cluster-creation",
	}

	// Add RAID mount paths if specified
	if raidMountPath, ok := job.Config["raid_mount_path"].(string); ok && raidMountPath != "" {
		args = append(args, "--raid-mount-path", raidMountPath)
	}
	if raidName, ok := job.Config["raid_name"].(string); ok && raidName != "" {
		args = append(args, "--raid-name", raidName)
	}

	// Log the training configuration
	log.Printf("Starting mindlet training:")
	log.Printf("  Source Model: %s", sourceModelID)
	log.Printf("  Target Model: %s", targetModelID)
	log.Printf("  Model Name: %s", modelName)
	log.Printf("  Node Count: %d", nodeCount)
	log.Printf("  Checkpoint Path: %s", checkpointPath)
	log.Printf("  Dataset Path: %s", datasetPath)
	log.Printf("  Output Path: %s", outputPath)

	// Execute the training command
	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		job.mu.Lock()
		job.Status = JobStatusError
		job.LastError = fmt.Errorf("failed to start training: %w\nOutput: %s", err, string(output))
		job.mu.Unlock()
		return job.LastError
	}

	// Training started successfully
	job.mu.Lock()
	job.Status = JobStatusRunning
	job.mu.Unlock()

	log.Printf("Training job %s started successfully with %d nodes", job.ID, nodeCount)

	// Monitor training progress in a separate goroutine
	go m.monitorTrainingProgress(job)

	return nil
}

// Add training progress monitor
func (m *OrchestrationManager) monitorTrainingProgress(job *OrchestrationJob) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Check training status using mindlet train status command
			status, err := m.checkTrainingStatus(context.Background(), job.ID)
			if err != nil {
				log.Printf("Error checking training status for job %s: %v", job.ID, err)
				continue
			}

			job.mu.Lock()
			// Update job with latest status
			if statusStr, ok := status["status"].(string); ok {
				switch statusStr {
				case "completed":
					job.Status = JobStatusStopped
					now := time.Now()
					job.StoppedAt = &now
					job.mu.Unlock()
					log.Printf("Training job %s completed", job.ID)
					return
				case "error", "failed":
					job.Status = JobStatusError
					if errMsg, ok := status["error"].(string); ok {
						job.LastError = fmt.Errorf(errMsg)
					}
					now := time.Now()
					job.StoppedAt = &now
					job.mu.Unlock()
					log.Printf("Training job %s failed", job.ID)
					return
				case "running":
					// Continue monitoring
					job.mu.Unlock()
				default:
					job.mu.Unlock()
				}
			} else {
				job.mu.Unlock()
			}
		}
	}
}

// Add training status checker
func (m *OrchestrationManager) checkTrainingStatus(ctx context.Context, jobID string) (map[string]interface{}, error) {
	args := []string{
		"mindlet", "train", "status",
		"--skip-cluster-creation",
	}

	log.Printf("Checking training status for job %s", jobID)
	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		// Status command might fail if no training is running
		log.Printf("Training status check failed: %v, output: %s", err, string(output))
		return map[string]interface{}{
			"status": "not_running",
			"job_id": jobID,
		}, nil
	}

	// Parse the JSON output from mindlet train status
	var status map[string]interface{}
	if err := json.Unmarshal(output, &status); err != nil {
		log.Printf("Failed to parse training status JSON: %v, output: %s", err, string(output))
		return map[string]interface{}{
			"status": "error",
			"error":  "failed to parse status",
			"job_id": jobID,
		}, nil
	}

	return status, nil
}
