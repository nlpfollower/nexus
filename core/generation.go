package core

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// GenerationStream represents an active model generation stream
type GenerationStream interface {
	SendResponse(resp ModelResponse) bool
	ResponseChan() <-chan ModelResponse
	Stop()
}

// baseGenerationStream provides a basic implementation of GenerationStream
type baseGenerationStream struct {
	responseChan chan ModelResponse
	ctx          context.Context
	cancelFunc   context.CancelFunc
	once         sync.Once
	closed       atomic.Bool
}

func newBaseGenerationStream(ctx context.Context) (*baseGenerationStream, context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	return &baseGenerationStream{
		responseChan: make(chan ModelResponse, 10000),
		ctx:          ctx,
		cancelFunc:   cancel,
	}, ctx
}

func (s *baseGenerationStream) ResponseChan() <-chan ModelResponse {
	return s.responseChan
}

func (s *baseGenerationStream) SendResponse(resp ModelResponse) bool {
	if s.closed.Load() {
		return false
	}

	// CRITICAL FIX: Use blocking send with context check
	// This prevents dropping messages when the buffer is full
	select {
	case <-s.ctx.Done():
		return false
	case s.responseChan <- resp:
		return true
	case <-time.After(30 * time.Second):
		// Timeout after 30 seconds to prevent permanent blocking
		log.Printf("WARNING: Response channel blocked for 30 seconds, likely consumer is stuck")
		return false
	}
}

func (s *baseGenerationStream) Stop() {
	s.once.Do(func() {
		s.closed.Store(true)
		s.cancelFunc()

		// Drain the channel before closing to prevent sender goroutine blocking
		go func() {
			for range s.responseChan {
				// Discard remaining messages
			}
		}()

		// Give a moment for any pending sends to complete
		time.Sleep(100 * time.Millisecond)
		close(s.responseChan)
	})
}
