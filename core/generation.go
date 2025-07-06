package core

import (
	"context"
	"sync"
	"sync/atomic"
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
	cancelFunc   context.CancelFunc
	once         sync.Once
	closed       atomic.Bool
}

func newBaseGenerationStream(ctx context.Context) (*baseGenerationStream, context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	return &baseGenerationStream{
		responseChan: make(chan ModelResponse, 1000),
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

	select {
	case s.responseChan <- resp:
		return true
	default:
		return false
	}
}

func (s *baseGenerationStream) Stop() {
	s.once.Do(func() {
		s.closed.Store(true)
		s.cancelFunc()
		close(s.responseChan)
	})
}
