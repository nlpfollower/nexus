package core

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/stretchr/testify/require"
)

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

	// Create a mock connection
	connectionID := "test-conn-" + uuid.New().String()

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
	userID := db.NewDigest([]byte("test-user"))
	requestID := db.NewDigest([]byte(uuid.New().String()))
	jobID := fmt.Sprintf("test-training-job-%d", time.Now().Unix())

	trainReq := &TrainingRequest{
		JobID:          jobID,
		UserID:         userID,
		SourceModelID:  "llama-8b-base",
		TargetModelID:  "llama-8b-trained-test",
		CheckpointPath: "/mnt/cold/contents/dcp/llama-8b-base/checkpoint",
		OutputPath:     "/mnt/cold/contents/dcp/llama-8b-trained-test/checkpoint",
		Dataset:        string(datasetJSON),
		ModelSize:      "8b",
		LearningRate:   0.0001,
		BatchSize:      32,
		NumEpochs:      3,
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
	statusCheckCount := 0
	maxStatusChecks := 120

	for statusCheckCount < maxStatusChecks {
		statusCheckCount++

		// Create status request
		statusRequestID := db.NewDigest([]byte(uuid.New().String()))
		statusReq := &TrainingStatusRequest{
			JobID: jobID,
		}

		statusRequest := &Request{
			RequestID:    statusRequestID,
			Type:         RequestTypeTrainingStatus,
			ConnectionID: connectionID,
			Status:       RequestStatusPending,
			Data:         statusReq,
			CreatedAt:    time.Now(),
		}

		// Process status request
		err = nexus.processRequest(statusRequest)
		require.NoError(t, err)

		// Get the job from training manager to check status
		job, err := nexus.trainingMgr.GetJobStatus(jobID)
		require.NoError(t, err)

		t.Logf("Training status check %d/%d: Status=%s, Progress=%.2f%%",
			statusCheckCount, maxStatusChecks, job.Status, job.Progress*100)

		// Check if training has started processing
		if job.Status == "processing_dataset" || job.Status == "processed_dataset" ||
			job.Status == "starting_training" || job.Status == "training" {
			t.Log("Training job is progressing successfully")

			// For test purposes, we'll stop here since actual training would take too long
			// In a real scenario, you would wait for completion
			break
		}

		if job.Status == "error" {
			t.Fatalf("Training job failed with error: %s", job.Error)
		}

		// Wait before next check
		time.Sleep(5 * time.Second)
	}

	// Verify the job was created and started processing
	job, err := nexus.trainingMgr.GetJobStatus(jobID)
	require.NoError(t, err)
	require.NotNil(t, job)
	require.Equal(t, jobID, job.JobID)
	require.Equal(t, "llama-8b-base", job.SourceModelID)
	require.Equal(t, "llama-8b-trained-test", job.TargetModelID)
	require.Equal(t, 1, job.NodeCount, "8B model should use 1 node")

	// Log final status
	t.Logf("Training job %s final status: %s (Progress: %.2f%%)",
		jobID, job.Status, job.Progress*100)

	// Verify dataset was created
	if job.DatasetPath != "" {
		t.Logf("Dataset was processed and saved to: %s", job.DatasetPath)
	}

	// Clean shutdown
	t.Log("Test completed, shutting down nexus...")
}

// Test training with 70B model to verify node count
func TestNexusTraining70BModel(t *testing.T) {
	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found, skipping test")
	}

	// Create nexus
	cfg := &Config{Port: 20000} // Use a different test port
	nexus, err := NewNexus(cfg)
	require.NoError(t, err)

	// Start nexus
	err = nexus.Start()
	require.NoError(t, err)
	defer nexus.Stop()

	// Wait for nexus to be ready
	time.Sleep(2 * time.Second)

	// Create a simple dataset for 70B model test
	dataset := TrainingDataset{
		ContextMessages: []Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "What is machine learning?"},
			{Role: "assistant", Content: "Machine learning is a subset of artificial intelligence..."},
		},
		TrainingPrompt: "General knowledge Q&A",
	}

	datasetJSON, err := json.Marshal(dataset)
	require.NoError(t, err)

	// Create training request for 70B model
	userID := db.NewDigest([]byte("test-user"))
	requestID := db.NewDigest([]byte(uuid.New().String()))
	jobID := fmt.Sprintf("test-70b-job-%d", time.Now().Unix())

	trainReq := &TrainingRequest{
		JobID:          jobID,
		UserID:         userID,
		SourceModelID:  "llama-70b-base",
		TargetModelID:  "llama-70b-trained-test",
		CheckpointPath: "/mnt/cold/contents/dcp/llama-70b-base/checkpoint",
		OutputPath:     "/mnt/cold/contents/dcp/llama-70b-trained-test/checkpoint",
		Dataset:        string(datasetJSON),
		ModelSize:      "70B", // Test uppercase
		LearningRate:   0.0001,
		BatchSize:      16,
		NumEpochs:      1,
	}

	// Create internal request
	req := &Request{
		RequestID:    requestID,
		Type:         RequestTypeTraining,
		ConnectionID: "test-conn-70b",
		Status:       RequestStatusPending,
		Data:         trainReq,
		CreatedAt:    time.Now(),
	}

	// Process the training request
	t.Log("Submitting 70B model training request...")
	err = nexus.processRequest(req)
	require.NoError(t, err)

	// Wait for processing to start
	time.Sleep(3 * time.Second)

	// Verify the job was created with correct node count
	job, err := nexus.trainingMgr.GetJobStatus(jobID)
	require.NoError(t, err)
	require.NotNil(t, job)
	require.Equal(t, 2, job.NodeCount, "70B model should use 2 nodes")

	t.Logf("70B training job created with %d nodes (correct!)", job.NodeCount)
}
