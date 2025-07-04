package core

import (
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/stretchr/testify/require"
)

// waitForVLLMReady waits for the mindlet server to have VLLM ready
func waitForVLLMReady(endpoint string, timeout time.Duration) error {
	client := &http.Client{Timeout: 10 * time.Second}
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		// Check health endpoint
		resp, err := client.Get(endpoint + "/health")
		if err == nil {
			defer resp.Body.Close()

			if resp.StatusCode == http.StatusOK {
				var health map[string]interface{}
				if err := json.NewDecoder(resp.Body).Decode(&health); err == nil {
					// Check if VLLM is running
					if vllmInfo, ok := health["vllm"].([]interface{}); ok && len(vllmInfo) > 0 {
						return nil // VLLM is ready
					}
				}
			}
		}

		time.Sleep(5 * time.Second)
	}

	return fmt.Errorf("timeout waiting for VLLM to be ready")
}

func TestNexusTrainingWithDataset(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	// Create nexus
	cfg := &Config{Port: 19999} // Use a test port
	nexus, err := NewNexus(cfg)
	require.NoError(t, err)

	// Start nexus
	err = nexus.Start()
	require.NoError(t, err)
	defer nexus.Stop()

	// Wait for nexus to be ready
	time.Sleep(2 * time.Second)

	// Use empty connection ID - responses will be dropped silently
	connectionID := ""

	// Create user and model ID
	userID := db.NewDigest([]byte("test-user"))
	modelID := "llama-8b"
	checkpointPath := "/mnt/cold/contents/dcp/llama-8b/checkpoint"

	// First, ensure the inference server is started and model is loaded
	t.Log("Starting inference server and loading model...")

	// Make an inference request to trigger model loading and VLLM startup
	inferRequestID := db.NewDigest([]byte(uuid.New().String()))
	inferReq := &InferenceRequest{
		UserID:         userID,
		ModelID:        modelID,
		Messages:       []Message{{Role: "user", Content: "Hello, test"}},
		CheckpointPath: checkpointPath,
		ModelSize:      "8B", // Must be uppercase
	}

	inferRequest := &Request{
		RequestID:    inferRequestID,
		Type:         RequestTypeInference,
		ConnectionID: connectionID,
		Status:       RequestStatusPending,
		Data:         inferReq,
		CreatedAt:    time.Now(),
	}

	// Process the inference request
	err = nexus.processRequest(inferRequest)
	require.NoError(t, err)

	// Stop the inference immediately after starting it
	// We only needed to trigger model loading, not get the actual response
	time.Sleep(1 * time.Second) // Give it a moment to start
	nexus.stopGeneration(inferRequestID)

	// Wait for the orchestration job to get the endpoint
	var endpoint string
	checkCount := 0
	maxChecks := 60 // 5 minutes max

	for checkCount < maxChecks {
		checkCount++

		// Get running inference jobs
		inferenceJobs := nexus.orchestrationMgr.GetJobsByType(JobTypeInference)
		for _, job := range inferenceJobs {
			job.mu.RLock()
			if job.Status == JobStatusRunning && job.Endpoint != "" {
				endpoint = job.Endpoint
			}
			job.mu.RUnlock()

			if endpoint != "" {
				break
			}
		}

		if endpoint != "" {
			t.Logf("Found inference endpoint after %d checks: %s", checkCount, endpoint)
			break
		}

		time.Sleep(5 * time.Second)
	}

	require.NotEmpty(t, endpoint, "Failed to get inference endpoint")

	// Now wait for VLLM to be ready within the mindlet server
	t.Log("Waiting for VLLM to be ready within mindlet...")
	err = waitForVLLMReady(endpoint, 3*time.Minute)
	require.NoError(t, err, "VLLM failed to start")

	// Stop the inference generation that we started just to load the model
	// This prevents the streaming responses from continuing
	if gen, exists := nexus.activeGenerations.Get(inferRequestID.String()); exists {
		gen.stream.Stop()
		nexus.activeGenerations.Remove(inferRequestID.String())
		t.Log("Stopped initial inference generation")
	}

	// Give a moment for cleanup
	time.Sleep(2 * time.Second)

	t.Log("VLLM is ready, proceeding with training dataset creation...")

	// Create a rich dataset similar to dataset_manager_test.go
	contextMessages := []Message{
		{Role: "system", Content: "You are an AI researcher specializing in machine learning and neural networks."},
	}

	// Training prompts that will generate diverse content
	trainingPrompts := []string{
		"Explain the transformer architecture in detail, including self-attention mechanisms and positional encoding.",
		"Describe the process of training a large language model, from data preparation to fine-tuning.",
		"What are the key differences between supervised, unsupervised, and reinforcement learning?",
		"Explain gradient descent and its variants (SGD, Adam, RMSprop) with their trade-offs.",
		"How does batch normalization work and why is it important in deep neural networks?",
		"Describe the concept of transfer learning and its applications in modern AI.",
		"What is the vanishing gradient problem and how do techniques like LSTM and GRU address it?",
		"Explain the role of regularization techniques like dropout and L1/L2 regularization.",
	}

	// For this test, we'll simulate the dataset content rather than making actual inference calls
	// In a real scenario, you would make inference requests to generate content
	for i, prompt := range trainingPrompts {
		// Add the user prompt
		contextMessages = append(contextMessages, Message{
			Role:    "user",
			Content: prompt,
		})

		// Add a simulated assistant response
		// In production, this would come from actual model inference
		simulatedResponse := fmt.Sprintf("This is a detailed response about: %s. [Simulated content for test purposes - in production this would be actual model-generated content that thoroughly addresses the topic with technical depth and examples.]", prompt)

		contextMessages = append(contextMessages, Message{
			Role:    "assistant",
			Content: simulatedResponse,
		})

		t.Logf("Added training example %d/%d", i+1, len(trainingPrompts))
	}

	// Create the training dataset
	dataset := TrainingDataset{
		ContextMessages: contextMessages,
		TrainingPrompt:  "Advanced machine learning concepts and architectures",
	}

	// Marshal dataset to JSON
	datasetJSON, err := json.Marshal(dataset)
	require.NoError(t, err)

	// Create training request
	requestID := db.NewDigest([]byte(uuid.New().String()))
	jobID := fmt.Sprintf("test-training-job-%d", time.Now().Unix())

	trainReq := &TrainingRequest{
		JobID:          jobID,
		UserID:         userID,
		SourceModelID:  modelID,
		TargetModelID:  "llama-8b-trained-test",
		CheckpointPath: checkpointPath,
		OutputPath:     "/mnt/cold/contents/dcp/llama-8b-trained-test/checkpoint",
		Dataset:        string(datasetJSON),
		ModelSize:      "8B", // Must be uppercase
	}

	// Create internal request
	req := &Request{
		RequestID:    requestID,
		Type:         RequestTypeTraining,
		ConnectionID: connectionID,
		Status:       RequestStatusPending,
		Data:         trainReq,
		CreatedAt:    time.Now(),
	}

	// Process the training request
	t.Log("Submitting training request to nexus...")
	err = nexus.processRequest(req)
	require.NoError(t, err)

	// Wait for initial response
	time.Sleep(2 * time.Second)

	// Check training status periodically
	// We want to see the job progress through multiple stages:
	// processing_dataset -> processed_dataset -> starting_training -> training
	statusCheckCount := 0
	maxStatusChecks := 60 // 5 minutes max
	datasetProcessed := false
	trainingStarted := false

	for statusCheckCount < maxStatusChecks {
		statusCheckCount++

		// Get the job from training manager to check status
		job, err := nexus.trainingMgr.GetJobStatus(jobID)
		require.NoError(t, err)

		t.Logf("Training status check %d/%d: Status=%s, Progress=%.2f%%",
			statusCheckCount, maxStatusChecks, job.Status, job.Progress*100)

		// Track progress through stages
		if job.Status == "processed_dataset" {
			datasetProcessed = true
			t.Log("Dataset processing completed successfully")
		}

		if job.Status == "starting_training" || job.Status == "training" {
			trainingStarted = true
			t.Log("Training has started successfully")
			// We can stop here for the test
			break
		}

		if job.Status == "error" {
			t.Fatalf("Training job failed with error: %s", job.Error)
		}

		// Wait before next check
		time.Sleep(5 * time.Second)
	}

	// Verify the job progressed through expected stages
	require.True(t, datasetProcessed || trainingStarted,
		"Job should have at least processed the dataset or started training")

	// Final verification
	job, err := nexus.trainingMgr.GetJobStatus(jobID)
	require.NoError(t, err)
	require.NotNil(t, job)
	require.Equal(t, jobID, job.JobID)
	require.Equal(t, modelID, job.SourceModelID)
	require.Equal(t, "llama-8b-trained-test", job.TargetModelID)
	require.Equal(t, 1, job.NodeCount, "8B model should use 1 node")

	// Log final status
	t.Logf("Training job %s final status: %s (Progress: %.2f%%)",
		jobID, job.Status, job.Progress*100)

	// Verify dataset was created if we got that far
	if datasetProcessed && job.DatasetPath != "" {
		t.Logf("Dataset was processed and saved to: %s", job.DatasetPath)
	}

	// Clean shutdown
	t.Log("Test completed, shutting down nexus...")
}
