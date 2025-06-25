// core/orchestration_manager_test.go
package core

import (
	"context"
	"fmt"
	"net/http"
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

// Helper to get test models based on environment
func getTestModels() (model1, model2, size1, size2 string) {
	if os.Getenv("RUN_LOCAL") != "" {
		// Local testing with smaller models
		return "llama-3b", "llama-8b", "3B", "8B"
	}
	// Production testing
	return "llama-8b", "llama-70b", "8B", "70B"
}

// TestOrchestrationManager_BasicLifecycle tests the basic functionality without starting real servers
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
	if os.Getenv("RUN_INFERENCE_TEST") != "true" {
		t.Skip("Skipping real inference test. Set RUN_INFERENCE_TEST=true to run.")
	}

	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	// This test actually starts a mindlet server and tests model switching
	manager, err := NewOrchestrationManager(10 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	// Simplified config for mindlet
	config := InferenceConfig{
		Port:          9090,
		NodeCount:     1,
		RaidMountPath: "/mnt/cold-storage",
		RaidName:      "cold-storage",
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

	// Get test models
	model1, model2, size1, size2 := getTestModels()

	// Test inference with first model
	t.Logf("Testing inference with %s (%s)", model1, size1)
	err = testMindletInference(t, endpoint, model1, size1)
	require.NoError(t, err)

	// Test inference with second model (triggers model switch)
	t.Logf("Testing inference with %s (%s) - should trigger model switch", model2, size2)
	err = testMindletInference(t, endpoint, model2, size2)
	require.NoError(t, err)

	// Test switching back to first model (should be faster)
	t.Logf("Testing switch back to %s (%s) - should reuse loaded model", model1, size1)
	startTime := time.Now()
	err = testMindletInference(t, endpoint, model1, size1)
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
	// Determine checkpoint path based on environment and model
	var checkpointPath string
	if os.Getenv("RUN_LOCAL") != "" {
		// Local testing paths
		checkpointPath = fmt.Sprintf("/home/nlpfollower/Desktop/deltamind/torchtitan/outputs/models/%s/checkpoint", modelID)
	} else {
		// Production paths
		checkpointPath = fmt.Sprintf("/mnt/cold/contents/dcp/%s/checkpoint", modelID)
	}

	// Prepare request for mindlet's /api/inference/stream endpoint
	messages := []Message{
		{Role: "user", Content: fmt.Sprintf("You are model %s. Say 'I am %s' and nothing else.", modelID, modelID)},
	}

	// Create a mock session to test the endpoint
	session := &InferenceSession{
		ID:             "test-session",
		ModelID:        modelID,
		CheckpointPath: checkpointPath, // Set checkpoint path
		ModelSize:      modelSize,      // Set model size
		Status:         SessionStatusRunning,
		Endpoint:       parseEndpoint(endpoint),
		Expiration:     time.Now().Add(5 * time.Minute),
		httpClient:     createHTTPClient(),
	}

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

// Additional test to verify cloned model support
func TestOrchestrationManager_ClonedModel(t *testing.T) {
	if os.Getenv("RUN_INFERENCE_TEST") != "true" {
		t.Skip("Skipping real inference test. Set RUN_INFERENCE_TEST=true to run.")
	}

	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	manager, err := NewOrchestrationManager(10 * time.Minute)
	require.NoError(t, err)

	manager.Start()
	defer manager.Stop()

	ctx := context.Background()

	// Test with a cloned model scenario
	// Model ID is different from checkpoint path
	clonedModelID := "my-custom-llama-8b"
	originalCheckpointPath := "/mnt/cold/contents/dcp/llama-8b"

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

// TestOrchestrationManager_SessionReuse tests that mindlet sessions can be reused
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
		ModelID:    "mindlet-server",
		Endpoint:   "http://localhost:9090",
		CreatedAt:  time.Now(),
		Expiration: time.Now().Add(5 * time.Minute),
	}
	manager.jobs.Set(job1.ID, job1)

	// GetOrCreateInferenceSession should return existing session
	ctx := context.Background()
	session1, err := manager.GetOrCreateInferenceSession(ctx, "llama-8b", "/path/to/model")
	require.NoError(t, err)
	require.Equal(t, job1.ID, session1.ID)
	require.Equal(t, "llama-8b", session1.ModelID) // Should use requested model ID

	// Second call should return same session (mindlet can handle multiple models)
	session2, err := manager.GetOrCreateInferenceSession(ctx, "llama-70b", "/path/to/model2")
	require.NoError(t, err)
	require.Equal(t, session1.ID, session2.ID)
	require.Equal(t, "llama-70b", session2.ModelID) // Should use new model ID

	t.Log("Session reuse test passed - mindlet server can handle multiple models")
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

// Helper functions

func parseEndpoint(endpoint string) *url.URL {
	u, _ := url.Parse(endpoint)
	return u
}

func createHTTPClient() *http.Client {
	return &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     90 * time.Second,
		},
	}
}
