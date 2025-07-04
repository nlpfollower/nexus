// core/orchestration_manager_test.go
package core

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Helper to check if orchestration directory exists
func orchestrationDirExists() bool {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return false
	}

	orchestrationDir := filepath.Join(homeDir, "orchestration")
	_, err = os.Stat(orchestrationDir)
	return err == nil
}

// TestOrchestrationManager_BasicLifecycle tests the basic functionality without starting real servers
func TestOrchestrationManager_BasicLifecycle(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	// Create manager with 30 minute expiration
	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	// Test basic job tracking without actually starting servers
	jobID := "test-job-1"
	job := &OrchestrationJob{
		ID:         jobID,
		Type:       JobTypeInference,
		Status:     JobStatusRunning,
		ModelID:    "test-model",
		CreatedAt:  time.Now(),
		Expiration: time.Now().Add(5 * time.Minute),
		Endpoint:   "http://localhost:9090",
	}

	manager.jobs.Set(jobID, job)

	// Test GetJob
	retrievedJob, exists := manager.GetJob(jobID)
	require.True(t, exists)
	require.Equal(t, jobID, retrievedJob.ID)

	// Test GetJobsByType
	inferenceJobs := manager.GetJobsByType(JobTypeInference)
	require.Len(t, inferenceJobs, 1)
	require.Equal(t, jobID, inferenceJobs[0].ID)

	// Test GetRunningJobs
	runningJobs := manager.GetRunningJobs()
	require.Len(t, runningJobs, 1)

	// Test ExtendJob
	err = manager.ExtendJob(jobID, 10*time.Minute)
	require.NoError(t, err)

	job.mu.RLock()
	newExpiration := job.Expiration
	job.mu.RUnlock()
	require.True(t, newExpiration.After(time.Now().Add(9*time.Minute)))
}

// TestOrchestrationManager_MindletInference tests real mindlet server with model switching
func TestOrchestrationManager_MindletInference(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	// This test actually starts a mindlet server and tests model switching
	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	// Simplified config for mindlet
	config := InferenceConfig{
		Port:          9090,
		NodeCount:     1,
		RaidMountPath: "/mnt/cold",
		RaidName:      "cold-new",
	}

	t.Log("Starting mindlet server...")

	// Start the mindlet server
	job, err := manager.StartInferenceJob(ctx, "mindlet-test", config)
	require.NoError(t, err)

	// Wait for server to be ready
	var endpoint string
	require.Eventually(t, func() bool {
		job.mu.RLock()
		status := job.Status
		endpoint = job.Endpoint
		job.mu.RUnlock()

		if status == JobStatusRunning && endpoint != "" {
			t.Logf("Mindlet server is running at %s", endpoint)
			return true
		}

		if status == JobStatusError {
			t.Log("Server failed to start")
			return true // Stop waiting
		}

		return false
	}, 6*time.Minute, 10*time.Second)

	// Check final status
	job.mu.RLock()
	finalStatus := job.Status
	job.mu.RUnlock()

	require.Equal(t, JobStatusRunning, finalStatus, "Server should be running")
	require.NotEmpty(t, endpoint, "Endpoint should be set")

	// Test inference with first model (8B)
	t.Log("Testing inference with llama-8b (8B)")
	err = testMindletInference(t, endpoint, "llama-8b", "8B")
	require.NoError(t, err)

	// Test inference with second model (70B) - triggers model switch
	t.Log("Testing inference with llama-70b (70B) - should trigger model switch")
	err = testMindletInference(t, endpoint, "llama-70b", "70B")
	require.NoError(t, err)

	// Test switching back to first model (should be faster)
	t.Log("Testing switch back to llama-8b (8B) - should reuse loaded model")
	startTime := time.Now()
	err = testMindletInference(t, endpoint, "llama-8b", "8B")
	require.NoError(t, err)
	switchTime := time.Since(startTime)
	t.Logf("Model switch took %v", switchTime)

	// Stop the server
	t.Log("Stopping mindlet server...")
	err = manager.StopJob(ctx, job.ID)
	require.NoError(t, err)

	// Verify it's stopped
	require.Eventually(t, func() bool {
		job.mu.RLock()
		defer job.mu.RUnlock()
		return job.Status == JobStatusStopped
	}, 30*time.Second, 2*time.Second)

	t.Log("Mindlet inference test completed successfully!")
}

// testMindletInference sends a test request to the mindlet server
func testMindletInference(t *testing.T, endpoint, modelID, modelSize string) error {
	// Production checkpoint path with /checkpoint suffix
	checkpointPath := fmt.Sprintf("/mnt/cold/contents/dcp/%s/checkpoint", modelID)

	// Prepare request for mindlet's /api/inference/stream endpoint
	messages := []Message{
		{Role: "user", Content: fmt.Sprintf("You are model %s. Say 'I am %s' and nothing else.", modelID, modelID)},
	}

	// Create a session using the constructor
	session := NewInferenceSession("test-session", modelID, 30*time.Minute, false)
	session.SetCheckpointPath(checkpointPath)
	session.SetModelSize(modelSize)
	session.Status = SessionStatusRunning
	session.Endpoint = parseEndpoint(endpoint)

	// Process inference using the session
	ctx := context.Background()
	stream, err := session.ProcessInference(ctx, messages)
	if err != nil {
		return fmt.Errorf("failed to process inference: %w", err)
	}

	// Collect streamed responses
	var fullResponse strings.Builder
	responseCount := 0
	timeout := time.After(30 * time.Second)
	done := make(chan bool)

	go func() {
		for resp := range stream.ResponseChan() {
			if resp.Error != nil {
				t.Logf("Stream error: %v", resp.Error)
				continue
			}

			if resp.Content != "" {
				fullResponse.WriteString(resp.Content)
				responseCount++
				if responseCount <= 5 {
					t.Logf("Chunk %d: %q", responseCount, resp.Content)
				}
			}
		}
		done <- true
	}()

	select {
	case <-done:
		t.Log("Streaming completed")
	case <-timeout:
		t.Log("Streaming timeout reached")
		stream.Stop()
	}

	finalResponse := fullResponse.String()
	t.Logf("Full response (%d chunks): %s", responseCount, finalResponse)

	// Verify we got a response
	if responseCount == 0 {
		return fmt.Errorf("no response received from model %s", modelID)
	}

	// Verify the response mentions the model (basic validation)
	if !strings.Contains(strings.ToLower(finalResponse), strings.ToLower(modelID)) {
		t.Logf("Warning: Response doesn't mention model ID %s", modelID)
	}

	return nil
}

// TestOrchestrationManager_SessionReuse tests that mindlet sessions can be reused
func TestOrchestrationManager_SessionReuse(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	// Manually create a running job
	job1 := &OrchestrationJob{
		ID:         "job-1",
		Type:       JobTypeInference,
		Status:     JobStatusRunning,
		ModelID:    "mindlet-server",
		Endpoint:   "http://localhost:9090",
		CreatedAt:  time.Now(),
		Expiration: time.Now().Add(5 * time.Minute),
	}
	manager.jobs.Set(job1.ID, job1)

	// GetOrCreateInferenceSession should return existing session
	ctx := context.Background()
	session1, err := manager.GetOrCreateInferenceSession(ctx, "llama-8b", "/mnt/cold/contents/dcp/llama-8b/checkpoint")
	require.NoError(t, err)
	require.Equal(t, job1.ID, session1.ID)
	require.Equal(t, "llama-8b", session1.ModelID)
	require.Equal(t, "/mnt/cold/contents/dcp/llama-8b/checkpoint", session1.CheckpointPath)

	// Second call should return same session (mindlet can handle multiple models)
	session2, err := manager.GetOrCreateInferenceSession(ctx, "llama-70b", "/mnt/cold/contents/dcp/llama-70b/checkpoint")
	require.NoError(t, err)
	require.Equal(t, session1.ID, session2.ID)
	require.Equal(t, "llama-70b", session2.ModelID)
	require.Equal(t, "/mnt/cold/contents/dcp/llama-70b/checkpoint", session2.CheckpointPath)

	t.Log("Session reuse test passed - mindlet server can handle multiple models")
}

// TestOrchestrationManager_ClonedModel tests support for cloned models
func TestOrchestrationManager_ClonedModel(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	// Test with a cloned model scenario
	// Model ID is different from checkpoint path
	clonedModelID := "my-custom-llama-8b"
	originalCheckpointPath := "/mnt/cold/contents/dcp/llama-8b/checkpoint"

	// Start inference session with cloned model
	session, err := manager.GetOrCreateInferenceSession(ctx, clonedModelID, originalCheckpointPath)
	require.NoError(t, err)
	require.NotNil(t, session)

	// Verify session has correct properties
	require.Equal(t, clonedModelID, session.ModelID)
	require.Equal(t, originalCheckpointPath, session.CheckpointPath)
	require.Equal(t, SessionStatusRunning, session.Status)

	// Test inference with the cloned model
	messages := []Message{
		{Role: "user", Content: "Hello, I am testing a cloned model. Please respond."},
	}

	stream, err := session.ProcessInference(ctx, messages)
	require.NoError(t, err)

	// Collect response
	var response strings.Builder
	timeout := time.After(30 * time.Second)
	done := make(chan bool)

	go func() {
		for resp := range stream.ResponseChan() {
			if resp.Error != nil {
				t.Logf("Stream error: %v", resp.Error)
				continue
			}
			if resp.Content != "" {
				response.WriteString(resp.Content)
			}
		}
		done <- true
	}()

	select {
	case <-done:
		t.Log("Response received successfully")
	case <-timeout:
		t.Error("Timeout waiting for response")
		stream.Stop()
	}

	require.NotEmpty(t, response.String(), "Should have received a response")
	t.Logf("Cloned model response: %s", response.String())

	// Clean up
	err = manager.StopJob(ctx, session.ID)
	require.NoError(t, err)
}

// TestOrchestrationManager_Expiration tests job expiration
func TestOrchestrationManager_Expiration(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	// Create manager with very short expiration
	manager, err := NewOrchestrationManager(100 * time.Millisecond)
	require.NoError(t, err)

	// Don't start the manager to avoid automatic expiration
	// Just test the expiration logic manually

	// Create an expired job
	job := &OrchestrationJob{
		ID:         "expired-job",
		Type:       JobTypeInference,
		Status:     JobStatusRunning,
		ModelID:    "test-model",
		CreatedAt:  time.Now().Add(-1 * time.Minute),
		Expiration: time.Now().Add(-30 * time.Second), // Already expired
	}

	manager.jobs.Set(job.ID, job)

	// Manually trigger expiration check
	manager.checkExpirations()

	// Job should be marked for expiration
	job.mu.RLock()
	status := job.Status
	job.mu.RUnlock()

	// Since we can't actually stop a non-existent job, it might be in error or expired state
	require.Contains(t, []OrchestrationJobStatus{JobStatusExpired, JobStatusError, JobStatusStopping}, status)
}

// TestOrchestrationManager_TrainingNodeCount tests that training uses appropriate node counts
func TestOrchestrationManager_TrainingNodeCount(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	testCases := []struct {
		name          string
		sourceModelID string
		targetModelID string
		expectedNodes int
	}{
		{
			name:          "8B model training",
			sourceModelID: "llama-8b-u123-c1",
			targetModelID: "llama-8b-u123-c1-t1",
			expectedNodes: 1,
		},
		{
			name:          "70B model training",
			sourceModelID: "llama-70b-u456-c1",
			targetModelID: "llama-70b-u456-c1-t1",
			expectedNodes: 2,
		},
		{
			name:          "3B model training",
			sourceModelID: "llama-3b-custom",
			targetModelID: "llama-3b-custom-t1",
			expectedNodes: 1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()

			// Test node count determination
			actualNodes := manager.determineNodeCountFromModel(tc.sourceModelID)
			require.Equal(t, tc.expectedNodes, actualNodes,
				"Model %s should use %d nodes, got %d", tc.sourceModelID, tc.expectedNodes, actualNodes)

			// Test training job creation with proper config
			trainingConfig := map[string]interface{}{
				"source_model_id": tc.sourceModelID,
				"target_model_id": tc.targetModelID,
				"checkpoint_path": fmt.Sprintf("/mnt/cold/contents/dcp/%s/checkpoint", tc.sourceModelID),
				"output_path":     fmt.Sprintf("/mnt/cold/contents/dcp/%s/checkpoint", tc.targetModelID),
				"dataset_path":    fmt.Sprintf("/mnt/cold/contents/datasets/dataset-%s", tc.targetModelID),
				"model_name":      tc.targetModelID,
				"node_count":      tc.expectedNodes, // Explicitly set node count
			}

			// Create training job (this would normally call mindlet train)
			job, err := manager.StartTrainingJob(ctx, tc.targetModelID, trainingConfig)
			require.NoError(t, err)
			require.NotNil(t, job)

			// Verify job properties
			require.Equal(t, JobTypeTraining, job.Type)
			require.Equal(t, tc.targetModelID, job.ModelID)

			// Verify node count is in config
			if nodeCount, ok := job.Config["node_count"].(int); ok {
				require.Equal(t, tc.expectedNodes, nodeCount)
			}

			// Wait a moment for job to initialize
			time.Sleep(100 * time.Millisecond)

			// Check job status
			jobStatus, err := manager.GetJobStatus(ctx, job.ID)
			require.NoError(t, err)
			require.Equal(t, job.ID, jobStatus.ID)

			// Clean up - stop the job
			err = manager.StopJob(ctx, job.ID)
			require.NoError(t, err)

			// Wait for stop to complete
			require.Eventually(t, func() bool {
				job.mu.RLock()
				defer job.mu.RUnlock()
				return job.Status == JobStatusStopped || job.Status == JobStatusError
			}, 10*time.Second, 500*time.Millisecond)
		})
	}
}

// TestOrchestrationManager_TrainingCommand tests the actual mindlet train command generation
func TestOrchestrationManager_TrainingCommand(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	testCases := []struct {
		name        string
		config      map[string]interface{}
		expectNodes int
	}{
		{
			name: "8B model config",
			config: map[string]interface{}{
				"source_model_id": "llama-8b-u123-c1",
				"target_model_id": "llama-8b-u123-c1-t1",
				"checkpoint_path": "/mnt/cold/contents/dcp/llama-8b-u123-c1/checkpoint",
				"output_path":     "/mnt/cold/contents/dcp/llama-8b-u123-c1-t1/checkpoint",
				"dataset_path":    "/mnt/cold/contents/datasets/dataset-llama-8b-u123-c1-t1",
				"model_name":      "llama-8b-u123-c1-t1",
			},
			expectNodes: 1,
		},
		{
			name: "70B model config",
			config: map[string]interface{}{
				"source_model_id": "llama-70b-u456-c2",
				"target_model_id": "llama-70b-u456-c2-t1",
				"checkpoint_path": "/mnt/cold/contents/dcp/llama-70b-u456-c2/checkpoint",
				"output_path":     "/mnt/cold/contents/dcp/llama-70b-u456-c2-t1/checkpoint",
				"dataset_path":    "/mnt/cold/contents/datasets/dataset-llama-70b-u456-c2-t1",
				"model_name":      "llama-70b-u456-c2-t1",
			},
			expectNodes: 2,
		},
		{
			name: "Explicit node count override",
			config: map[string]interface{}{
				"source_model_id": "custom-model",
				"target_model_id": "custom-model-t1",
				"checkpoint_path": "/mnt/cold/contents/dcp/custom-model/checkpoint",
				"output_path":     "/mnt/cold/contents/dcp/custom-model-t1/checkpoint",
				"dataset_path":    "/mnt/cold/contents/datasets/dataset-custom-model-t1",
				"model_name":      "custom-model-t1",
				"node_count":      3, // Explicit override
			},
			expectNodes: 3,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test node count determination
			actualNodes := manager.getNodeCountFromConfig(tc.config)
			require.Equal(t, tc.expectNodes, actualNodes)

			// Create a mock job to test command generation
			job := &OrchestrationJob{
				ID:         "test-job",
				Type:       JobTypeTraining,
				Status:     JobStatusPending,
				ModelID:    tc.config["target_model_id"].(string),
				Config:     tc.config,
				CreatedAt:  time.Now(),
				Expiration: time.Now().Add(time.Hour),
			}

			// Test that the training process would use the correct node count
			// We can't actually run the command, but we can verify the logic
			sourceModelID := tc.config["source_model_id"].(string)
			targetModelID := tc.config["target_model_id"].(string)
			checkpointPath := tc.config["checkpoint_path"].(string)
			outputPath := tc.config["output_path"].(string)
			datasetPath := tc.config["dataset_path"].(string)
			modelName := tc.config["model_name"].(string)
			nodeCount := manager.getNodeCountFromConfig(tc.config)

			// Verify all components are present
			require.Equal(t, tc.expectNodes, nodeCount)
			require.NotEmpty(t, sourceModelID)
			require.NotEmpty(t, targetModelID)
			require.NotEmpty(t, checkpointPath)
			require.NotEmpty(t, outputPath)
			require.NotEmpty(t, datasetPath)
			require.NotEmpty(t, modelName)

			t.Logf("Training config validated:")
			t.Logf("  Source: %s", sourceModelID)
			t.Logf("  Target: %s", targetModelID)
			t.Logf("  Nodes: %d", nodeCount)
			t.Logf("  Checkpoint: %s", checkpointPath)
			t.Logf("  Dataset: %s", datasetPath)
			t.Logf("  Output: %s", outputPath)
		})
	}
}

// TestOrchestrationManager_TrainingStatusMonitoring tests training status monitoring
func TestOrchestrationManager_TrainingStatusMonitoring(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	// Create a training job
	trainingConfig := map[string]interface{}{
		"source_model_id": "llama-8b-test",
		"target_model_id": "llama-8b-test-t1",
		"checkpoint_path": "/mnt/cold/contents/dcp/llama-8b-test/checkpoint",
		"output_path":     "/mnt/cold/contents/dcp/llama-8b-test-t1/checkpoint",
		"dataset_path":    "/mnt/cold/contents/datasets/dataset-llama-8b-test-t1",
		"model_name":      "llama-8b-test-t1",
		"node_count":      1,
	}

	job, err := manager.StartTrainingJob(ctx, "llama-8b-test-t1", trainingConfig)
	require.NoError(t, err)

	// Test status monitoring
	// Note: This will likely fail in a test environment since we don't have actual training running
	// But we can test that the status monitoring logic works
	status, err := manager.GetTrainingStatus(ctx, "llama-8b-test-t1")

	// The status check might fail if no training is actually running, which is expected in tests
	if err != nil {
		t.Logf("Expected error in test environment: %v", err)
		require.Contains(t, status["status"].(string), "not_running")
	} else {
		// If status check succeeds, verify it returns valid data
		require.Contains(t, status, "status")
		require.Contains(t, status, "job_id")

		statusStr := status["status"].(string)
		require.Contains(t, []string{"running", "completed", "error", "not_running"}, statusStr)
	}

	// Clean up
	err = manager.StopJob(ctx, job.ID)
	require.NoError(t, err)
}

// TestOrchestrationManager_MultipleTrainingJobs tests running multiple training jobs
func TestOrchestrationManager_MultipleTrainingJobs(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	// Create multiple training jobs with different node counts
	jobs := make([]*OrchestrationJob, 0)

	configs := []map[string]interface{}{
		{
			"source_model_id": "llama-8b-multi-1",
			"target_model_id": "llama-8b-multi-1-t1",
			"checkpoint_path": "/mnt/cold/contents/dcp/llama-8b-multi-1/checkpoint",
			"output_path":     "/mnt/cold/contents/dcp/llama-8b-multi-1-t1/checkpoint",
			"dataset_path":    "/mnt/cold/contents/datasets/dataset-llama-8b-multi-1-t1",
			"model_name":      "llama-8b-multi-1-t1",
		},
		{
			"source_model_id": "llama-70b-multi-2",
			"target_model_id": "llama-70b-multi-2-t1",
			"checkpoint_path": "/mnt/cold/contents/dcp/llama-70b-multi-2/checkpoint",
			"output_path":     "/mnt/cold/contents/dcp/llama-70b-multi-2-t1/checkpoint",
			"dataset_path":    "/mnt/cold/contents/datasets/dataset-llama-70b-multi-2-t1",
			"model_name":      "llama-70b-multi-2-t1",
		},
	}

	expectedNodeCounts := []int{1, 2} // 8B uses 1 node, 70B uses 2 nodes

	// Start all jobs
	for i, config := range configs {
		job, err := manager.StartTrainingJob(ctx, config["target_model_id"].(string), config)
		require.NoError(t, err)
		jobs = append(jobs, job)

		// Verify node count is determined correctly
		nodeCount := manager.getNodeCountFromConfig(config)
		require.Equal(t, expectedNodeCounts[i], nodeCount)
	}

	// Wait for jobs to initialize
	time.Sleep(500 * time.Millisecond)

	// Verify all jobs are tracked
	trainingJobs := manager.GetJobsByType(JobTypeTraining)
	require.GreaterOrEqual(t, len(trainingJobs), len(jobs))

	// Clean up all jobs
	for _, job := range jobs {
		err := manager.StopJob(ctx, job.ID)
		require.NoError(t, err)
	}

	// Wait for all jobs to stop
	for _, job := range jobs {
		require.Eventually(t, func() bool {
			job.mu.RLock()
			defer job.mu.RUnlock()
			return job.Status == JobStatusStopped || job.Status == JobStatusError
		}, 10*time.Second, 500*time.Millisecond)
	}
}

// TestOrchestrationManager_EdgeCases tests edge cases in training
func TestOrchestrationManager_EdgeCases(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	manager, err := NewOrchestrationManager(30 * time.Minute)
	require.NoError(t, err)

	t.Run("Unknown model defaults to 1 node", func(t *testing.T) {
		nodeCount := manager.determineNodeCountFromModel("unknown-experimental-model")
		require.Equal(t, 1, nodeCount)
	})

	t.Run("Empty model ID defaults to 1 node", func(t *testing.T) {
		nodeCount := manager.determineNodeCountFromModel("")
		require.Equal(t, 1, nodeCount)
	})

	t.Run("Case insensitive model detection", func(t *testing.T) {
		testCases := []struct {
			modelID  string
			expected int
		}{
			{"LLAMA-8B", 1},
			{"Llama-70B", 2},
			{"llama-8B-CUSTOM", 1},
			{"LLAMA-70b-experimental", 2},
		}

		for _, tc := range testCases {
			nodeCount := manager.determineNodeCountFromModel(tc.modelID)
			require.Equal(t, tc.expected, nodeCount,
				"Model %s should use %d nodes", tc.modelID, tc.expected)
		}
	})

	t.Run("Explicit node count override", func(t *testing.T) {
		config := map[string]interface{}{
			"node_count":      5,          // Explicit override
			"source_model_id": "llama-8b", // Would normally be 1 node
		}

		nodeCount := manager.getNodeCountFromConfig(config)
		require.Equal(t, 5, nodeCount, "Explicit node count should override model-based detection")
	})

	t.Run("Invalid explicit node count falls back to model detection", func(t *testing.T) {
		config := map[string]interface{}{
			"node_count":      "invalid", // Invalid type
			"source_model_id": "llama-70b",
		}

		nodeCount := manager.getNodeCountFromConfig(config)
		require.Equal(t, 2, nodeCount, "Should fall back to model-based detection when explicit count is invalid")
	})
}

// Helper functions

func parseEndpoint(endpoint string) *url.URL {
	u, _ := url.Parse(endpoint)
	return u
}
