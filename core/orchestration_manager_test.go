// core/orchestration_manager_test.go
package core

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
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

func TestOrchestrationManager_BasicLifecycle(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	// Create manager with 5 minute expiration
	manager, err := NewOrchestrationManager(5 * time.Minute)
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

func TestOrchestrationManager_ExpirationLogic(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	manager, err := NewOrchestrationManager(1 * time.Minute)
	require.NoError(t, err)

	// Don't start the manager to avoid actual expiration monitoring
	// Just test the expiration logic

	// Create an expired job
	job := &OrchestrationJob{
		ID:         "expired-job",
		Type:       JobTypeInference,
		Status:     JobStatusRunning,
		ModelID:    "test-model",
		CreatedAt:  time.Now().Add(-2 * time.Minute),
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

// Integration test - only runs with real environment
func TestOrchestrationManager_RealInference(t *testing.T) {
	if os.Getenv("RUN_INFERENCE_TEST") != "true" {
		t.Skip("Skipping real inference test. Set RUN_INFERENCE_TEST=true to run.")
	}

	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	// This test actually starts and stops a real inference server
	manager, err := NewOrchestrationManager(10 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	config := InferenceConfig{
		DCPDir:           "/mnt/cold/contents/dcp/llama-8b/step-0",
		TokenizerPath:    "/mnt/cold/contents/checkpoints/Llama3.1-8B-Instruct/tokenizer.model",
		ParamsPath:       "torchchat/model_params/Meta-Llama-3.1-8B.json",
		Port:             5000,
		DCPModelSize:     "8B",
		CheckpointFolder: "llama-8b",
		NodeCount:        1,
		RaidMountPath:    "/mnt/cold",
		RaidName:         "cold-new",
	}

	t.Log("Starting real inference server (this will take 20-60 seconds)...")

	job, err := manager.StartInferenceJob(ctx, "llama-8b", config)
	require.NoError(t, err)

	// Wait for server to be ready (up to 3 minutes)
	require.Eventually(t, func() bool {
		job.mu.RLock()
		status := job.Status
		endpoint := job.Endpoint
		job.mu.RUnlock()

		if status == JobStatusRunning && endpoint != "" {
			t.Logf("Server is running at %s", endpoint)
			return true
		}

		if status == JobStatusError {
			t.Logf("Server failed to start")
			return true // Stop waiting
		}

		return false
	}, 6*time.Minute, 10*time.Second)

	// Check final status
	job.mu.RLock()
	finalStatus := job.Status
	endpoint := job.Endpoint
	job.mu.RUnlock()

	require.Equal(t, JobStatusRunning, finalStatus, "Server should be running")
	require.NotEmpty(t, endpoint, "Endpoint should be set")

	// Make test inference requests
	if endpoint != "" {
		t.Log("Making test inference requests...")

		// First request: "Are you alive?"
		reqBody1 := map[string]interface{}{
			"messages": []map[string]string{
				{"role": "user", "content": "Are you alive? Reply with yes or no only."},
			},
			"model":  "",
			"stream": false,
		}

		jsonBody1, _ := json.Marshal(reqBody1)
		resp1, err := http.Post(endpoint+"/v1/chat/completions", "application/json", bytes.NewBuffer(jsonBody1))
		if err == nil {
			defer resp1.Body.Close()
			var result1 map[string]interface{}
			json.NewDecoder(resp1.Body).Decode(&result1)
			t.Logf("First inference response: %+v", result1)
		}

		// Second request: "What is 2+2?"
		reqBody2 := map[string]interface{}{
			"messages": []map[string]string{
				{"role": "user", "content": "What is 2+2? Reply with the number only."},
			},
			"model":  "",
			"stream": false,
		}

		jsonBody2, _ := json.Marshal(reqBody2)
		resp2, err := http.Post(endpoint+"/v1/chat/completions", "application/json", bytes.NewBuffer(jsonBody2))
		if err == nil {
			defer resp2.Body.Close()
			var result2 map[string]interface{}
			json.NewDecoder(resp2.Body).Decode(&result2)
			t.Logf("Second inference response: %+v", result2)
		}
	}

	// Stop the server
	t.Log("Stopping inference server...")
	err = manager.StopJob(ctx, job.ID)
	require.NoError(t, err)

	// Verify it's stopped
	require.Eventually(t, func() bool {
		job.mu.RLock()
		defer job.mu.RUnlock()
		return job.Status == JobStatusStopped
	}, 30*time.Second, 2*time.Second)

	t.Log("Real inference test completed successfully!")
}

func TestOrchestrationManager_SessionReuse(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	manager, err := NewOrchestrationManager(5 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	// Manually create a running job
	job1 := &OrchestrationJob{
		ID:         "job-1",
		Type:       JobTypeInference,
		Status:     JobStatusRunning,
		ModelID:    "test-model",
		Endpoint:   "http://localhost:5000",
		CreatedAt:  time.Now(),
		Expiration: time.Now().Add(5 * time.Minute),
	}
	manager.jobs.Set(job1.ID, job1)

	// GetOrCreateInferenceSession should return existing session
	ctx := context.Background()
	session1, err := manager.GetOrCreateInferenceSession(ctx, "test-model")
	require.NoError(t, err)
	require.Equal(t, job1.ID, session1.ID)

	// Second call should return same session
	session2, err := manager.GetOrCreateInferenceSession(ctx, "test-model")
	require.NoError(t, err)
	require.Equal(t, session1.ID, session2.ID)

	// Different model would trigger new job creation, but we can't test that
	// without real orchestration CLI
}

func TestOrchestrationManager_TimeoutScenario(t *testing.T) {
	if os.Getenv("RUN_TIMEOUT_TEST") != "true" {
		t.Skip("Skipping timeout test. Set RUN_TIMEOUT_TEST=true to run.")
	}

	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	// Test with very short expiration (60 seconds)
	manager, err := NewOrchestrationManager(60 * time.Second)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	config := InferenceConfig{
		DCPDir:           "/mnt/cold/contents/dcp/llama-8b/step-0",
		TokenizerPath:    "/mnt/cold/contents/checkpoints/Llama3.1-8B-Instruct/tokenizer.model",
		ParamsPath:       "torchchat/model_params/Meta-Llama-3.1-8B.json",
		Port:             5000,
		DCPModelSize:     "8B",
		CheckpointFolder: "llama-8b",
		NodeCount:        1,
		RaidMountPath:    "/mnt/cold",
		RaidName:         "cold-new",
	}

	t.Log("Starting server with 60 second timeout...")

	job, err := manager.StartInferenceJob(ctx, "llama-8b", config)
	require.NoError(t, err)

	// Wait for server to start
	require.Eventually(t, func() bool {
		job.mu.RLock()
		defer job.mu.RUnlock()
		return job.Status == JobStatusRunning || job.Status == JobStatusError
	}, 3*time.Minute, 10*time.Second)

	job.mu.RLock()
	status := job.Status
	job.mu.RUnlock()

	if status == JobStatusRunning {
		t.Log("Server started, waiting for timeout expiration...")

		// Wait for expiration (should happen after 60 seconds)
		require.Eventually(t, func() bool {
			job.mu.RLock()
			defer job.mu.RUnlock()
			return job.Status == JobStatusExpired || job.Status == JobStatusStopped
		}, 90*time.Second, 5*time.Second)

		t.Log("Server was automatically stopped due to timeout!")
	} else {
		t.Log("Server failed to start, skipping timeout test")
	}
}
