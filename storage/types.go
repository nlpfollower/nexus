package storage

import (
	"github.com/nlpfollower/deltamind/database/db"
	"time"
)

// Add nexus-specific types here, for example:

type UserLimits struct {
	UserID        db.Digest `json:"user_id"`
	MessageCount  int       `json:"message_count"`
	LastResetTime time.Time `json:"last_reset_time"`
}
