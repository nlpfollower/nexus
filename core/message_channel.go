package core

import (
	"fmt"
	"sync/atomic"
	"time"
)

// MessageChannel handles request queuing and response streaming
type MessageChannel struct {
	requests  chan *Request
	responses chan *Response
	closed    atomic.Bool
}

func NewMessageChannel(requestBufferSize, responseBufferSize int) *MessageChannel {
	return &MessageChannel{
		requests:  make(chan *Request, requestBufferSize),
		responses: make(chan *Response, responseBufferSize),
	}
}

func (mc *MessageChannel) EnqueueRequest(wrapped *WrappedRequest, connID string) error {
	if mc.closed.Load() {
		return fmt.Errorf("message channel is closed")
	}

	req, err := UnwrapRequest(wrapped, connID)
	if err != nil {
		return fmt.Errorf("failed to unwrap request: %w", err)
	}

	select {
	case mc.requests <- req:
		return nil
	default:
		return fmt.Errorf("request channel full")
	}
}

func (mc *MessageChannel) EnqueueResponse(resp *WrappedResponse, connID string) error {
	if mc.closed.Load() {
		return fmt.Errorf("message channel is closed")
	}

	mc.responses <- &Response{
		Response:     resp,
		ConnectionID: connID,
		Timestamp:    time.Now(),
	}
	return nil
}

func (mc *MessageChannel) GetRequestChannel() <-chan *Request {
	return mc.requests
}

func (mc *MessageChannel) GetResponseChannel() <-chan *Response {
	return mc.responses
}

func (mc *MessageChannel) Close() {
	if mc.closed.CompareAndSwap(false, true) {
		close(mc.requests)
		close(mc.responses)
	}
}

// IsClosed returns whether the channel has been closed
func (mc *MessageChannel) IsClosed() bool {
	return mc.closed.Load()
}
