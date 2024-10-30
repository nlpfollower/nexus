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
	RequestTypeInference RequestType = "INFERENCE"
	RequestTypeSetUser   RequestType = "SET_USER"

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
	UserID   db.Digest `json:"user_id"`
	ModelID  db.Digest `json:"model_id"`
	Messages []Message `json:"messages"`
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
