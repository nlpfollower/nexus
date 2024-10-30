// core/request_tracker.go
package core

import (
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"time"
)

// RequestTracker manages active requests and their states in memory
type RequestTracker struct {
	requests   *utils.ConcurrentMap[string, *Request] // key is RequestID.String()
	users      *utils.ConcurrentMap[string, struct{}] // key is UserID.String()
	userLimits *utils.ConcurrentMap[string, *UserLimits]
}

type UserLimits struct {
	MessageCount  int
	LastResetTime time.Time
}

func NewRequestTracker() *RequestTracker {
	return &RequestTracker{
		requests:   utils.NewConcurrentMap[string, *Request](),
		users:      utils.NewConcurrentMap[string, struct{}](),
		userLimits: utils.NewConcurrentMap[string, *UserLimits](),
	}
}

// AddRequest adds a new request to tracking
func (rt *RequestTracker) AddRequest(req *Request) {
	rt.requests.Set(req.RequestID.String(), req)
}

// AddUser adds a new user ID to tracking
func (rt *RequestTracker) AddUser(userID db.Digest) {
	rt.users.Set(userID.String(), struct{}{})
}

// GetRequest retrieves a request by ID
func (rt *RequestTracker) GetRequest(requestID db.Digest) (*Request, bool) {
	return rt.requests.Get(requestID.String())
}

// UpdateStatus updates a request's status
func (rt *RequestTracker) UpdateStatus(requestID db.Digest, status RequestStatus) bool {
	req, ok := rt.requests.Get(requestID.String())
	if !ok {
		return false
	}

	req.Status = status
	rt.requests.Set(requestID.String(), req)
	return true
}

// GetPendingRequests returns all requests with Pending status
func (rt *RequestTracker) GetPendingRequests() []*Request {
	var pending []*Request

	for _, req := range rt.requests.GetAll() {
		if req.Status == RequestStatusPending {
			pending = append(pending, req)
		}
	}

	return pending
}

// GetAllocatedRequests returns all requests with Allocated status
func (rt *RequestTracker) GetAllocatedRequests() []*Request {
	var allocated []*Request

	for _, req := range rt.requests.GetAll() {
		if req.Status == RequestStatusAllocated {
			allocated = append(allocated, req)
		}
	}

	return allocated
}

// RemoveRequest removes a completed/errored request from tracking
func (rt *RequestTracker) RemoveRequest(requestID db.Digest) {
	rt.requests.Remove(requestID.String())
}

// UserExists checks if a user ID exists in tracking
func (rt *RequestTracker) UserExists(userID db.Digest) bool {
	_, exists := rt.users.Get(userID.String())
	return exists
}

// SetUserLimits sets limits for a user
func (rt *RequestTracker) SetUserLimits(userID db.Digest, limits *UserLimits) {
	rt.userLimits.Set(userID.String(), limits)
}

// GetUserLimits gets current limits for a user
func (rt *RequestTracker) GetUserLimits(userID db.Digest) (*UserLimits, bool) {
	return rt.userLimits.Get(userID.String())
}
