// core/orchestration_manager_test.go
package core

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"io"
	"net"
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

func TestOrchestrationManager_StreamingInference(t *testing.T) {
	if os.Getenv("RUN_INFERENCE_TEST") != "true" {
		t.Skip("Skipping streaming inference test. Set RUN_INFERENCE_TEST=true to run.")
	}

	if !orchestrationDirExists() {
		t.Skip("Orchestration directory not found")
	}

	// Start the orchestration manager
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

	t.Log("Starting inference server for streaming test...")

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
			job.mu.RLock()
			lastErr := job.LastError
			job.mu.RUnlock()
			t.Logf("Server failed to start: %v", lastErr)
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

	// Now test streaming with the running server
	t.Log("Server ready, testing streaming...")

	// Get the session that was created
	session, err := manager.GetOrCreateInferenceSession(ctx, "llama-8b")
	require.NoError(t, err)
	require.NotNil(t, session)

	// Create messages for streaming test
	messages := []Message{
		{Role: "user", Content: "Tell me a very short story in 2-3 sentences."},
	}

	// Process inference with streaming
	stream, err := session.ProcessInference(ctx, messages)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Collect streamed responses
	var fullResponse strings.Builder
	responseCount := 0

	t.Log("Receiving streamed responses...")

	// Set a timeout for reading from the stream
	streamTimeout := time.After(30 * time.Second)
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
				// Only log first few chunks to avoid spam
				if responseCount <= 10 {
					t.Logf("Chunk %d: %q", responseCount, resp.Content)
				}
			}
		}
		done <- true
	}()

	// Wait for streaming to complete or timeout
	select {
	case <-done:
		t.Log("Streaming completed")
	case <-streamTimeout:
		t.Log("Streaming timeout reached")
		stream.Stop()
	}

	finalResponse := fullResponse.String()
	t.Logf("Full response (%d chunks): %s", responseCount, finalResponse)

	// Verify we got a response - even a single chunk is fine for now
	require.Greater(t, responseCount, 0, "Should receive at least one chunk")
	require.NotEmpty(t, finalResponse, "Should have received content")

	t.Logf("Streaming test completed successfully with %d chunks!", responseCount)

	t.Log("Streaming test completed successfully!")

	// Clean up - stop the server
	t.Log("Stopping inference server...")
	err = manager.StopJob(ctx, job.ID)
	require.NoError(t, err)

	// Wait for it to stop
	require.Eventually(t, func() bool {
		job.mu.RLock()
		defer job.mu.RUnlock()
		return job.Status == JobStatusStopped
	}, 30*time.Second, 2*time.Second)
}

func TestDirectStreamingHTTP(t *testing.T) {
	if os.Getenv("RUN_INFERENCE_TEST") != "true" {
		t.Skip("Skipping direct streaming test. Set RUN_INFERENCE_TEST=true to run.")
	}

	// This test assumes the server is already running at the endpoint
	endpoint := "http://192.168.172.13:5000" // Update this to match your server

	// Create request body
	reqBody := map[string]interface{}{
		"stream": true,
		"model":  "",
		"messages": []Message{
			{Role: "user", Content: "Say hello in 3 words"},
		},
	}

	jsonBody, err := json.Marshal(reqBody)
	require.NoError(t, err)

	// Create HTTP request
	req, err := http.NewRequest("POST", endpoint+"/v1/chat/completions", bytes.NewBuffer(jsonBody))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	// Send request
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	t.Logf("Response status: %s", resp.Status)
	t.Logf("Response headers: %v", resp.Header)

	// Read response body
	scanner := bufio.NewScanner(resp.Body)
	lineCount := 0
	chunkCount := 0

	for scanner.Scan() {
		line := scanner.Text()
		lineCount++
		t.Logf("Line %d: %q", lineCount, line)

		if strings.HasPrefix(line, "data:") {
			chunkCount++
			jsonData := strings.TrimPrefix(line, "data:")
			jsonData = strings.TrimSpace(jsonData)

			var data map[string]interface{}
			if err := json.Unmarshal([]byte(jsonData), &data); err != nil {
				t.Logf("Failed to parse JSON: %v", err)
			} else {
				t.Logf("Parsed chunk %d: %+v", chunkCount, data)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		t.Logf("Scanner error: %v", err)
	}

	t.Logf("Total lines: %d, chunks: %d", lineCount, chunkCount)
	require.Greater(t, chunkCount, 0, "Should have received at least one chunk")
}

func TestSessionDirectStreaming(t *testing.T) {
	if os.Getenv("RUN_INFERENCE_TEST") != "true" {
		t.Skip("Skipping direct session test. Set RUN_INFERENCE_TEST=true to run.")
	}

	// Create a session directly with a known endpoint
	endpoint, err := url.Parse("http://192.168.160.114:5000") // Update this IP
	require.NoError(t, err)

	session := &InferenceSession{
		ID:           "test-session",
		ModelID:      "test-model",
		Status:       SessionStatusRunning,
		Endpoint:     endpoint,
		Expiration:   time.Now().Add(5 * time.Minute),
		IsPersistent: false,
		httpClient:   &http.Client{Timeout: 60 * time.Second},
	}

	ctx := context.Background()
	messages := []Message{
		{Role: "user", Content: "Say hello in 3 words"},
	}

	// Process inference
	stream, err := session.ProcessInference(ctx, messages)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Collect responses
	var chunks []string
	responseCount := 0

	t.Log("Starting to read from stream...")

	// Use a timeout
	timeout := time.After(30 * time.Second)
	done := make(chan bool)

	go func() {
		for resp := range stream.ResponseChan() {
			if resp.Error != nil {
				t.Logf("Stream error: %v", resp.Error)
				continue
			}

			if resp.Content != "" {
				responseCount++
				chunks = append(chunks, resp.Content)
				t.Logf("Chunk %d: %q", responseCount, resp.Content)
			}
		}
		done <- true
	}()

	select {
	case <-done:
		t.Log("Stream completed")
	case <-timeout:
		t.Log("Stream timeout")
		stream.Stop()
	}

	t.Logf("Received %d chunks: %v", responseCount, chunks)
	require.Greater(t, responseCount, 0, "Should have received at least one chunk")
}

// Test to see how the raw stream data comes in
func TestRawStreamReading(t *testing.T) {
	if os.Getenv("RUN_INFERENCE_TEST") != "true" {
		t.Skip("Skipping raw stream test. Set RUN_INFERENCE_TEST=true to run.")
	}

	endpoint := "http://192.168.164.222:5000" // Update this IP

	// Create request
	reqBody := map[string]interface{}{
		"stream": true,
		"model":  "",
		"messages": []Message{
			{Role: "user", Content: "Count to 3"},
		},
	}

	jsonBody, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", endpoint+"/v1/chat/completions", bytes.NewBuffer(jsonBody))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	t.Log("Starting to read stream...")

	// List to collect all messages
	var messages []map[string]interface{}

	// Read SSE messages
	reader := bufio.NewReader(resp.Body)
	for {
		// Read lines until we get a complete SSE message
		var sseMessage strings.Builder
		hasData := false

		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					t.Log("Stream ended")
					goto done
				}
				t.Logf("Read error: %v", err)
				goto done
			}

			// Check if this line contains data
			if strings.HasPrefix(line, "data:") {
				hasData = true
			}

			sseMessage.WriteString(line)

			// SSE messages are separated by empty lines
			if line == "\n" && hasData {
				break
			}
		}

		// Process the SSE message
		fullMessage := sseMessage.String()
		lines := strings.Split(fullMessage, "\n")

		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "data:") {
				dataContent := strings.TrimSpace(strings.TrimPrefix(line, "data:"))

				// Skip [DONE] message
				if dataContent == "[DONE]" {
					t.Log("Received [DONE] signal")
					goto done
				}

				// Parse JSON
				var message map[string]interface{}
				if err := json.Unmarshal([]byte(dataContent), &message); err != nil {
					t.Logf("Failed to parse JSON: %v", err)
					continue
				}

				// Add to messages list
				messages = append(messages, message)

				// Log for debugging (only first few)
				if len(messages) <= 10 {
					t.Logf("Message %d: %+v", len(messages), message)
				}

				// Check for stop signal
				if choices, ok := message["choices"].([]interface{}); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]interface{}); ok {
						if finishReason, ok := choice["finish_reason"].(string); ok && finishReason == "stop" {
							t.Log("Received finish_reason: stop")
							goto done
						}
					}
				}
			}
		}
	}

done:
	// Print summary
	t.Logf("\n=== Summary ===")
	t.Logf("Total messages received: %d", len(messages))
	t.Log("\nAll messages:")
	for i, msg := range messages {
		t.Logf("Message %d: %+v", i+1, msg)
	}
}

func TestNexusE2E_SessionAndInference(t *testing.T) {
	if os.Getenv("RUN_E2E_TEST") != "true" {
		t.Skip("Skipping E2E test. Set RUN_E2E_TEST=true to run.")
	}

	// Get an available port for nexus
	listener, err := net.Listen("tcp", ":0")
	require.NoError(t, err)
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	cfg := &Config{
		Port: port,
	}

	// Start nexus
	nexus, err := NewNexus(cfg)
	require.NoError(t, err)

	err = nexus.Start()
	require.NoError(t, err)
	defer nexus.Stop()

	t.Logf("Nexus started on port %d", port)

	// Connect to nexus
	conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	require.NoError(t, err)
	defer conn.Close()

	encoder := json.NewEncoder(conn)
	decoder := json.NewDecoder(conn)

	// Step 1: Send SESSION request to start inference server
	t.Log("Step 1: Starting inference session...")

	// Use the actual model name that has a checkpoint
	modelID := "llama-8b"
	sessionReqID := db.NewDigest([]byte("session-request-1"))

	sessionReq := &SessionRequest{
		Action:  SessionActionStart,
		ModelID: modelID,
	}

	wrappedSessionReq, err := NewWrappedRequest(sessionReqID, sessionReq)
	require.NoError(t, err)

	err = encoder.Encode(wrappedSessionReq)
	require.NoError(t, err)

	// Wait for session response (should return immediately now)
	var sessionResp WrappedResponse
	err = decoder.Decode(&sessionResp)
	require.NoError(t, err)
	require.Equal(t, sessionReqID, sessionResp.RequestID)

	var sessionResponse SessionResponse
	err = json.Unmarshal(sessionResp.Data, &sessionResponse)
	require.NoError(t, err)
	require.Equal(t, ResponseStatusSuccess, sessionResponse.Status)
	require.NotEmpty(t, sessionResponse.SessionID)

	sessionID := sessionResponse.SessionID
	t.Logf("Session %s started (state: %s)", sessionID, sessionResponse.State)

	// Step 1b: Poll for session readiness
	t.Log("Step 1b: Waiting for session to be ready...")

	var sessionEndpoint string
	deadline := time.Now().Add(10 * time.Minute)
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	pollCount := 0
	for time.Now().Before(deadline) {
		pollCount++

		// Send status check request
		statusReqID := db.NewDigest([]byte(fmt.Sprintf("session-status-%d", pollCount)))
		statusReq := &SessionRequest{
			Action:    SessionActionStatus,
			SessionID: sessionID,
		}

		wrappedStatusReq, err := NewWrappedRequest(statusReqID, statusReq)
		require.NoError(t, err)

		err = encoder.Encode(wrappedStatusReq)
		require.NoError(t, err)

		// Read status response
		var statusResp WrappedResponse
		err = decoder.Decode(&statusResp)
		require.NoError(t, err)
		require.Equal(t, statusReqID, statusResp.RequestID)

		var statusResponse SessionResponse
		err = json.Unmarshal(statusResp.Data, &statusResponse)
		require.NoError(t, err)

		t.Logf("Poll %d: Session state = %s", pollCount, statusResponse.State)

		// Check if session is ready
		if statusResponse.State == string(JobStatusRunning) && statusResponse.Endpoint != "" {
			sessionEndpoint = statusResponse.Endpoint
			t.Logf("Session ready after %d polls! Endpoint: %s", pollCount, sessionEndpoint)
			break
		}

		if statusResponse.State == string(JobStatusError) {
			t.Fatalf("Session failed to start after %d polls", pollCount)
		}

		<-ticker.C
	}

	require.NotEmpty(t, sessionEndpoint, "Session should have an endpoint after initialization")

	// Give the server a moment to stabilize
	time.Sleep(2 * time.Second)

	// Step 2: Send INFERENCE request
	t.Log("Step 2: Sending inference request...")

	inferReqID := db.NewDigest([]byte("inference-request-1"))
	userID := db.NewDigest([]byte("test-user"))

	inferReq := &InferenceRequest{
		UserID:  userID,
		ModelID: modelID,
		Messages: []Message{
			{Role: "user", Content: "Count to 3 and say hello"},
		},
	}

	wrappedInferReq, err := NewWrappedRequest(inferReqID, inferReq)
	require.NoError(t, err)

	err = encoder.Encode(wrappedInferReq)
	require.NoError(t, err)

	// Step 3: Collect streaming responses
	t.Log("Step 3: Collecting streaming responses...")

	var responses []InferenceResponse
	var fullContent strings.Builder
	done := make(chan struct{})
	streamErr := make(chan error, 1)

	go func() {
		defer close(done)

		for {
			var resp WrappedResponse
			if err := decoder.Decode(&resp); err != nil {
				if !strings.Contains(err.Error(), "use of closed network connection") {
					streamErr <- fmt.Errorf("decode error: %w", err)
				}
				return
			}

			// Verify it's for our request
			if resp.RequestID != inferReqID {
				continue
			}

			var inferResp InferenceResponse
			if err := json.Unmarshal(resp.Data, &inferResp); err != nil {
				streamErr <- fmt.Errorf("unmarshal error: %w", err)
				return
			}

			responses = append(responses, inferResp)
			fullContent.WriteString(inferResp.Content)

			t.Logf("Response %d: Type=%s, Status=%s, Content=%q",
				len(responses), inferResp.Type, inferResp.Status, inferResp.Content)

			// Check if this is the final response
			if inferResp.Type == ResponseTypeFinal {
				t.Log("Received final response")
				return
			}

			// Check for errors
			if inferResp.Status == ResponseStatusError {
				streamErr <- fmt.Errorf("inference error: %s", inferResp.Content)
				return
			}
		}
	}()

	// Wait for streaming to complete
	select {
	case <-done:
		t.Log("Streaming completed successfully")
	case err := <-streamErr:
		t.Fatalf("Streaming error: %v", err)
	case <-time.After(30 * time.Second):
		t.Fatal("Timeout waiting for streaming responses")
	}

	// Verify results
	t.Logf("\n=== Results ===")
	t.Logf("Total responses: %d", len(responses))
	t.Logf("Full content: %q", fullContent.String())

	require.Greater(t, len(responses), 0, "Should have received at least one response")
	require.NotEmpty(t, fullContent.String(), "Should have received content")

	// Verify we got both partial and final responses
	hasPartial := false
	hasFinal := false
	for _, resp := range responses {
		if resp.Type == ResponseTypePartial {
			hasPartial = true
		}
		if resp.Type == ResponseTypeFinal {
			hasFinal = true
		}
	}
	require.True(t, hasPartial, "Should have received partial responses")
	require.True(t, hasFinal, "Should have received final response")

	// Step 4: Send another inference request to test session reuse
	t.Log("\nStep 4: Testing session reuse with second inference...")

	inferReqID2 := db.NewDigest([]byte("inference-request-2"))
	inferReq2 := &InferenceRequest{
		UserID:  userID,
		ModelID: modelID,
		Messages: []Message{
			{Role: "user", Content: "What is 2+2?"},
		},
	}

	wrappedInferReq2, err := NewWrappedRequest(inferReqID2, inferReq2)
	require.NoError(t, err)

	err = encoder.Encode(wrappedInferReq2)
	require.NoError(t, err)

	// Collect second response
	done2 := make(chan struct{})
	var secondResponse strings.Builder

	go func() {
		defer close(done2)

		for {
			var resp WrappedResponse
			if err := decoder.Decode(&resp); err != nil {
				return
			}

			if resp.RequestID != inferReqID2 {
				continue
			}

			var inferResp InferenceResponse
			json.Unmarshal(resp.Data, &inferResp)
			secondResponse.WriteString(inferResp.Content)

			if inferResp.Type == ResponseTypeFinal {
				return
			}
		}
	}()

	select {
	case <-done2:
		t.Logf("Second inference completed: %q", secondResponse.String())
	case <-time.After(15 * time.Second):
		t.Fatal("Timeout on second inference")
	}

	require.Contains(t, secondResponse.String(), "4", "Response should contain '4'")

	// Step 5: Stop the session
	t.Log("\nStep 5: Stopping the session...")

	sessionStopReqID := db.NewDigest([]byte("session-stop-request"))
	sessionStopReq := &SessionRequest{
		Action:    SessionActionStop,
		SessionID: sessionID,
	}

	wrappedStopReq, err := NewWrappedRequest(sessionStopReqID, sessionStopReq)
	require.NoError(t, err)

	err = encoder.Encode(wrappedStopReq)
	require.NoError(t, err)

	// Wait for stop response
	var stopResp WrappedResponse
	err = decoder.Decode(&stopResp)
	require.NoError(t, err)
	require.Equal(t, sessionStopReqID, stopResp.RequestID)

	var stopResponse SessionResponse
	err = json.Unmarshal(stopResp.Data, &stopResponse)
	require.NoError(t, err)
	require.Equal(t, ResponseStatusSuccess, stopResponse.Status)

	t.Log("Session stopped successfully")
	t.Log("\n✅ End-to-end test completed successfully!")
}
