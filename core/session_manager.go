// core/session_manager.go
package core

import (
	"fmt"
	"github.com/nlpfollower/deltamind/database/db"
	"github.com/nlpfollower/deltamind/orchestration/utils"
	"sync"
	"time"
)

// SessionType represents different types of infrastructure sessions
type SessionType string

const (
	SessionTypeInference SessionType = "INFERENCE"
	SessionTypeTraining  SessionType = "TRAINING"
)

// SessionStatus represents the current state of a session
type SessionStatus string

const (
	SessionStatusPending    SessionStatus = "PENDING"
	SessionStatusAllocating SessionStatus = "ALLOCATING"
	SessionStatusRunning    SessionStatus = "RUNNING"
	SessionStatusCompleted  SessionStatus = "COMPLETED"
)

// Session represents an active infrastructure session
type Session struct {
	ID     string
	Type   SessionType
	Status SessionStatus
	NodeID string // Reference to infrastructure node

	// Future fields for infrastructure support
	BatchSize    int
	BatchTimeout time.Duration
	StartTime    time.Time
	UpdateTime   time.Time
}

// SessionManager handles infrastructure-based model sessions
type SessionManager struct {
	sessions     *utils.ConcurrentMap[string, *Session]
	mu           sync.Mutex
	maxBatchSize int

	// Future fields
	startupTimeout time.Duration
	batchTimeout   time.Duration
}

func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions:       utils.NewConcurrentMap[string, *Session](),
		maxBatchSize:   10,
		startupTimeout: 2 * time.Minute,
		batchTimeout:   30 * time.Second,
	}
}

// GetOrCreateSession returns an existing session or creates a new one
func (sm *SessionManager) GetOrCreateSession(modelID db.Digest) (*Session, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// TODO: Infrastructure session management
	// This will be implemented when deltamind/orchestration support is added
	// Expected functionality:
	// 1. Check for existing session for this model
	// 2. If none exists or all are full, create new session
	// 3. Handle session startup (estimated 2min)
	// 4. Return session for request routing

	return nil, fmt.Errorf("infrastructure sessions not yet implemented")
}

// GetSession retrieves an existing session
func (sm *SessionManager) GetSession(sessionID string) (*Session, bool) {
	return sm.sessions.Get(sessionID)
}

// CloseSession gracefully shuts down a session
func (sm *SessionManager) CloseSession(sessionID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// TODO: Implement proper cleanup when infrastructure support is added
	// This will include:
	// 1. Finish processing current batch
	// 2. Release infrastructure resources
	// 3. Clean up session state

	sm.sessions.Remove(sessionID)
	return nil
}
