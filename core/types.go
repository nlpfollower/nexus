// core/types.go
package core

import (
	"encoding/json"
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"time"
)

type NexusRequest interface {
	NexusRequestType() RequestType
}

type NexusResponse interface {
	NexusResponseType() RequestType
}

type RequestType string
type RequestStatus string
type ResponseType string
type ResponseStatus string

const (
	RequestTypeInference      RequestType = "INFERENCE"
	RequestTypeSetUser        RequestType = "SET_USER"
	RequestTypeSession        RequestType = "SESSION"
	RequestTypeJobStatus      RequestType = "JOB_STATUS"
	RequestTypeTraining       RequestType = "TRAINING"
	RequestTypeTrainingStatus RequestType = "TRAINING_STATUS"

	RequestStatusPending    RequestStatus = "PENDING"
	RequestStatusAllocated  RequestStatus = "ALLOCATED"
	RequestStatusProcessing RequestStatus = "PROCESSING"
	RequestStatusCompleted  RequestStatus = "COMPLETED"
	RequestStatusError      RequestStatus = "ERROR"

	ResponseTypePartial ResponseType = "PARTIAL"
	ResponseTypeFinal   ResponseType = "FINAL"

	ResponseStatusSuccess ResponseStatus = "SUCCESS"
	ResponseStatusError   ResponseStatus = "ERROR"
)

// OrchestrationJobType represents the type of orchestration job
type OrchestrationJobType string

const (
	JobTypeInference OrchestrationJobType = "inference"
	JobTypeTraining  OrchestrationJobType = "training"
)

// OrchestrationJobStatus represents the status of a job
type OrchestrationJobStatus string

const (
	JobStatusPending      OrchestrationJobStatus = "pending"
	JobStatusInitializing OrchestrationJobStatus = "initializing"
	JobStatusRunning      OrchestrationJobStatus = "running"
	JobStatusStopping     OrchestrationJobStatus = "stopping"
	JobStatusStopped      OrchestrationJobStatus = "stopped"
	JobStatusError        OrchestrationJobStatus = "error"
	JobStatusExpired      OrchestrationJobStatus = "expired"
)

// Session status types (used by session.go)
type SessionStatus string

const (
	SessionStatusInitializing SessionStatus = "INITIALIZING"
	SessionStatusRunning      SessionStatus = "RUNNING"
	SessionStatusStopping     SessionStatus = "STOPPING"
	SessionStatusStopped      SessionStatus = "STOPPED"
	SessionStatusError        SessionStatus = "ERROR"
)

// Client-facing request types
type WrappedRequest struct {
	RequestID db.Digest       `json:"request_id"`
	Type      RequestType     `json:"type"`
	Data      json.RawMessage `json:"data"`
}

// Client-facing response types
type WrappedResponse struct {
	RequestID db.Digest       `json:"request_id"`
	Type      RequestType     `json:"type"`
	Data      json.RawMessage `json:"data"`
}

// Request implementations
type InferenceRequest struct {
	UserID         db.Digest `json:"user_id"`
	ModelID        string    `json:"model_id"`
	Messages       []Message `json:"messages"`
	CheckpointPath string    `json:"checkpoint_path,omitempty"`
	ModelSize      string    `json:"model_size,omitempty"` // NEW: Model size (3B, 8B, 70B, etc.)
}

func (r *InferenceRequest) NexusRequestType() RequestType {
	return RequestTypeInference
}

type InferenceResponse struct {
	Type    ResponseType   `json:"type"` // partial or final
	Content string         `json:"content"`
	Status  ResponseStatus `json:"status"`
}

func (r *InferenceResponse) NexusResponseType() RequestType {
	return RequestTypeInference
}

type SetUserRequest struct {
	UserID db.Digest `json:"user_id"`
}

func (r *SetUserRequest) NexusRequestType() RequestType {
	return RequestTypeSetUser
}

type SetUserResponse struct {
	Status ResponseStatus `json:"status"`
}

func (r *SetUserResponse) NexusResponseType() RequestType {
	return RequestTypeSetUser
}

// Session management types
type SessionAction string

const (
	SessionActionStart  SessionAction = "START"
	SessionActionStop   SessionAction = "STOP"
	SessionActionExtend SessionAction = "EXTEND"
	SessionActionStatus SessionAction = "STATUS" // New action for checking status
)

type SessionRequest struct {
	Action    SessionAction `json:"action"`
	ModelID   string        `json:"model_id,omitempty"`   // Required for START
	SessionID string        `json:"session_id,omitempty"` // Required for STOP, EXTEND, and STATUS
	Duration  string        `json:"duration,omitempty"`   // Optional for EXTEND, format like "30m"
}

func (r *SessionRequest) NexusRequestType() RequestType {
	return RequestTypeSession
}

type SessionResponse struct {
	Status    ResponseStatus `json:"status"`
	SessionID string         `json:"session_id,omitempty"`
	Endpoint  string         `json:"endpoint,omitempty"`
	State     string         `json:"state,omitempty"` // New field to indicate session state
	Error     string         `json:"error,omitempty"`
}

func (r *SessionResponse) NexusResponseType() RequestType {
	return RequestTypeSession
}

// Job status types
type JobStatusRequest struct {
	JobID string `json:"job_id"`
}

func (r *JobStatusRequest) NexusRequestType() RequestType {
	return RequestTypeJobStatus
}

type JobStatusResponse struct {
	Status  ResponseStatus `json:"status"`
	JobInfo *JobStatusInfo `json:"job_info,omitempty"`
	Error   string         `json:"error,omitempty"`
}

func (r *JobStatusResponse) NexusResponseType() RequestType {
	return RequestTypeJobStatus
}

// JobStatusInfo contains detailed status information about a job
type JobStatusInfo struct {
	ID         string                 `json:"id"`
	Type       OrchestrationJobType   `json:"type"`
	Status     OrchestrationJobStatus `json:"status"`
	ModelID    string                 `json:"model_id"`
	CreatedAt  time.Time              `json:"created_at"`
	StartedAt  *time.Time             `json:"started_at,omitempty"`
	StoppedAt  *time.Time             `json:"stopped_at,omitempty"`
	Expiration time.Time              `json:"expiration"`
	Endpoint   string                 `json:"endpoint,omitempty"`
	LiveStatus map[string]interface{} `json:"live_status,omitempty"`
}

type TrainingRequest struct {
	JobID          string    `json:"job_id"`
	UserID         db.Digest `json:"user_id"`
	SourceModelID  string    `json:"source_model_id"` // Model name
	TargetModelID  string    `json:"target_model_id"` // New model name
	CheckpointPath string    `json:"checkpoint_path"`
	OutputPath     string    `json:"output_path"`
	DatasetPath    string    `json:"dataset_path"`
	LearningRate   float64   `json:"learning_rate"`
	BatchSize      int       `json:"batch_size"`
	NumEpochs      int       `json:"num_epochs"`
}

func (r *TrainingRequest) NexusRequestType() RequestType {
	return RequestTypeTraining
}

type TrainingResponse struct {
	JobID  string `json:"job_id"`
	Status string `json:"status"`
}

func (r *TrainingResponse) NexusResponseType() RequestType {
	return RequestTypeTraining
}

type TrainingStatusRequest struct {
	JobID string `json:"job_id"`
}

func (r *TrainingStatusRequest) NexusRequestType() RequestType {
	return RequestTypeTrainingStatus
}

type TrainingStatusResponse struct {
	JobID       string     `json:"job_id"`
	Status      string     `json:"status"`
	Progress    float64    `json:"progress"`
	Error       string     `json:"error,omitempty"`
	StartedAt   time.Time  `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
}

func (r *TrainingStatusResponse) NexusResponseType() RequestType {
	return RequestTypeTrainingStatus
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Internal request wrapper
type Request struct {
	RequestID    db.Digest              `json:"request_id"` // From WrappedRequest
	Type         RequestType            `json:"type"`
	ConnectionID string                 `json:"connection_id"`
	Status       RequestStatus          `json:"status"`
	Data         NexusRequest           `json:"data"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
}

type Response struct {
	Response     *WrappedResponse
	ConnectionID string
	Timestamp    time.Time
}

// Generic helper functions
func NewWrappedRequest[T NexusRequest](requestID db.Digest, req T) (*WrappedRequest, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	return &WrappedRequest{
		RequestID: requestID,
		Type:      req.NexusRequestType(),
		Data:      data,
	}, nil
}

func NewWrappedResponse[T NexusResponse](requestID db.Digest, resp T) (*WrappedResponse, error) {
	data, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}

	return &WrappedResponse{
		RequestID: requestID,
		Type:      resp.NexusResponseType(),
		Data:      data,
	}, nil
}

// Helper to convert wrapped request to internal request
func UnwrapRequest(wrapped *WrappedRequest, connID string) (*Request, error) {
	switch wrapped.Type {
	case RequestTypeInference:
		var inferReq InferenceRequest
		if err := json.Unmarshal(wrapped.Data, &inferReq); err != nil {
			return nil, err
		}

		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeInference,
			ConnectionID: connID,
			Status:       RequestStatusPending,
			Data:         &inferReq,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeSetUser:
		var userReq SetUserRequest
		if err := json.Unmarshal(wrapped.Data, &userReq); err != nil {
			return nil, err
		}

		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeSetUser,
			ConnectionID: connID,
			Status:       RequestStatusPending,
			Data:         &userReq,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeSession:
		var sessionReq SessionRequest
		if err := json.Unmarshal(wrapped.Data, &sessionReq); err != nil {
			return nil, err
		}

		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeSession,
			ConnectionID: connID,
			Status:       RequestStatusPending,
			Data:         &sessionReq,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeJobStatus:
		var jobReq JobStatusRequest
		if err := json.Unmarshal(wrapped.Data, &jobReq); err != nil {
			return nil, err
		}

		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeJobStatus,
			ConnectionID: connID,
			Status:       RequestStatusPending,
			Data:         &jobReq,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeTraining:
		var trainReq TrainingRequest
		if err := json.Unmarshal(wrapped.Data, &trainReq); err != nil {
			return nil, err
		}
		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeTraining,
			ConnectionID: connID,
			Status:       RequestStatusPending,
			Data:         &trainReq,
			CreatedAt:    time.Now(),
		}, nil

	case RequestTypeTrainingStatus:
		var statusReq TrainingStatusRequest
		if err := json.Unmarshal(wrapped.Data, &statusReq); err != nil {
			return nil, err
		}
		return &Request{
			RequestID:    wrapped.RequestID,
			Type:         RequestTypeTrainingStatus,
			ConnectionID: connID,
			Status:       RequestStatusPending,
			Data:         &statusReq,
			CreatedAt:    time.Now(),
		}, nil
	}

	return nil, fmt.Errorf("unsupported request type: %s", wrapped.Type)
}

// Helper to unwrap responses for clients
func UnwrapResponse[T NexusResponse](wrapped *WrappedResponse) (*T, error) {
	var response T
	if err := json.Unmarshal(wrapped.Data, &response); err != nil {
		return nil, err
	}
	return &response, nil
}
