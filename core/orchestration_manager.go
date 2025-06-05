// core/orchestration_manager.go
package core

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
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
		close(m.done)

		// Stop all running jobs
		for _, job := range m.jobs.GetAll() {
			job.mu.RLock()
			isRunning := job.Status == JobStatusRunning || job.Status == JobStatusInitializing
			job.mu.RUnlock()

			if isRunning {
				ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
				if err := m.stopJob(ctx, job); err != nil {
					log.Printf("Error stopping job %s: %v", job.ID, err)
				}
				cancel()
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
func (m *OrchestrationManager) GetOrCreateInferenceSession(ctx context.Context, modelID string) (*InferenceSession, error) {
	// Check for existing running inference jobs for this model
	for _, job := range m.jobs.GetAll() {
		job.mu.RLock()
		isRunning := job.Status == JobStatusRunning && job.ModelID == modelID && job.Type == JobTypeInference
		endpoint := job.Endpoint
		jobID := job.ID
		expiration := job.Expiration
		job.mu.RUnlock()

		if isRunning && endpoint != "" {
			// Create a session wrapper for this job
			endpointURL, err := url.Parse(endpoint)
			if err != nil {
				continue
			}

			session := &InferenceSession{
				ID:           jobID,
				ModelID:      modelID,
				Status:       SessionStatusRunning,
				Endpoint:     endpointURL,
				Expiration:   expiration,
				IsPersistent: false,
				httpClient:   &http.Client{Timeout: 60 * time.Second},
			}

			log.Printf("Reusing existing session %s for model %s at %s", jobID, modelID, endpoint)
			return session, nil
		}
	}

	// No existing session, start a new inference job with default config
	config := InferenceConfig{
		DCPDir:           fmt.Sprintf("/mnt/cold/contents/dcp/%s/step-0", modelID),
		TokenizerPath:    "/mnt/cold/contents/checkpoints/Llama3.1-8B-Instruct/tokenizer.model",
		ParamsPath:       "torchchat/model_params/Meta-Llama-3.1-8B.json",
		Port:             5000,
		DCPModelSize:     "8B",
		CheckpointFolder: modelID,
		NodeCount:        1,
		RaidMountPath:    "/mnt/cold",
		RaidName:         "cold-new",
	}

	job, err := m.StartInferenceJob(ctx, modelID, config)
	if err != nil {
		return nil, fmt.Errorf("failed to start inference job: %w", err)
	}

	// Wait for the job to be ready
	readyCtx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-readyCtx.Done():
			return nil, fmt.Errorf("timeout waiting for inference job to be ready")
		case <-ticker.C:
			job.mu.RLock()
			status := job.Status
			endpoint := job.Endpoint
			lastError := job.LastError
			job.mu.RUnlock()

			if status == JobStatusError {
				return nil, fmt.Errorf("inference job failed: %w", lastError)
			}

			if status == JobStatusRunning && endpoint != "" {
				endpointURL, err := url.Parse(endpoint)
				if err != nil {
					return nil, fmt.Errorf("invalid endpoint URL: %w", err)
				}

				session := &InferenceSession{
					ID:           job.ID,
					ModelID:      modelID,
					Status:       SessionStatusRunning,
					Endpoint:     endpointURL,
					Expiration:   job.Expiration,
					IsPersistent: false,
					httpClient:   &http.Client{Timeout: 60 * time.Second},
				}

				log.Printf("Created new session %s for model %s at %s", job.ID, modelID, endpoint)
				return session, nil
			}
		}
	}
}

// executeOrchestrationCommand runs a command in the orchestration directory
func (m *OrchestrationManager) executeOrchestrationCommand(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "go", append([]string{"run", "main.go"}, args...)...)
	cmd.Dir = m.orchestrationDir
	return cmd.CombinedOutput()
}

func (m *OrchestrationManager) startInferenceProcess(ctx context.Context, job *OrchestrationJob, config InferenceConfig) error {
	job.mu.Lock()
	job.Status = JobStatusInitializing
	now := time.Now()
	job.StartedAt = &now
	job.mu.Unlock()

	// Build the command arguments
	args := []string{
		"mindlet", "inference",
		"--dcp-dir", config.DCPDir,
		"--tokenizer-path", config.TokenizerPath,
		"--params-path", config.ParamsPath,
		"--port", fmt.Sprintf("%d", config.Port),
		"--dcp-model-size", config.DCPModelSize,
		"--checkpoint-folder", config.CheckpointFolder,
		"--node-count", fmt.Sprintf("%d", config.NodeCount),
		"--skip-cluster-creation",
	}

	if config.RaidMountPath != "" {
		args = append(args, "--raid-mount-path", config.RaidMountPath)
	}
	if config.RaidName != "" {
		args = append(args, "--raid-name", config.RaidName)
	}

	// Execute the command
	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		job.mu.Lock()
		job.Status = JobStatusError
		job.LastError = fmt.Errorf("failed to start inference: %w\nOutput: %s", err, string(output))
		job.mu.Unlock()
		return job.LastError
	}

	// Wait for server to be ready with up to 3 minute timeout
	readyCtx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	log.Printf("Waiting for inference server to be ready (job %s)...", job.ID)

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-readyCtx.Done():
			job.mu.Lock()
			job.Status = JobStatusError
			job.LastError = fmt.Errorf("timeout waiting for inference server to be ready")
			job.mu.Unlock()
			return job.LastError

		case <-ticker.C:
			status, err := m.checkInferenceStatus(ctx)
			if err == nil {
				if serverStatus, ok := status["status"].(string); ok && serverStatus == "healthy" {
					// Server is ready
					job.mu.Lock()
					job.Status = JobStatusRunning
					if endpoint, ok := status["endpoint"].(string); ok {
						job.Endpoint = endpoint
					}
					job.mu.Unlock()

					log.Printf("Inference job %s started successfully", job.ID)
					return nil
				}
			}
			// Continue waiting
		}
	}
}

func (m *OrchestrationManager) stopJob(ctx context.Context, job *OrchestrationJob) error {
	job.mu.Lock()
	if job.Status == JobStatusStopped || job.Status == JobStatusStopping {
		job.mu.Unlock()
		return nil
	}

	job.Status = JobStatusStopping
	jobType := job.Type
	job.mu.Unlock()

	var err error
	switch jobType {
	case JobTypeInference:
		err = m.stopInferenceProcess(ctx, job)
	case JobTypeTraining:
		// TODO: Implement training stop logic
		err = fmt.Errorf("stopping training jobs not yet implemented")
	}

	job.mu.Lock()
	now := time.Now()
	job.StoppedAt = &now
	if err != nil {
		job.Status = JobStatusError
		job.LastError = err
	} else {
		job.Status = JobStatusStopped
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

func (m *OrchestrationManager) checkInferenceStatus(ctx context.Context) (map[string]interface{}, error) {
	args := []string{
		"mindlet", "inference", "status",
		"--skip-cluster-creation", "--silent-mode",
	}

	output, err := m.executeOrchestrationCommand(ctx, args...)
	if err != nil {
		// Status command might return non-zero exit code even with valid JSON output
		// Try to parse the output anyway
		var status map[string]interface{}
		if jsonErr := json.Unmarshal(output, &status); jsonErr == nil {
			return status, nil
		}
		return nil, fmt.Errorf("failed to get inference status: %w", err)
	}

	var status map[string]interface{}
	if err := json.Unmarshal(output, &status); err != nil {
		return nil, fmt.Errorf("failed to parse status output: %w", err)
	}

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
