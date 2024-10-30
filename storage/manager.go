package storage

import (
	"fmt"
	"os"
	"path/filepath"
)

type DatabaseManager struct {
	client *DatabaseClient
}

func NewDatabaseManager(dbPath string) (*DatabaseManager, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}
	client := NewDatabaseClient(dbPath)
	return &DatabaseManager{
		client: client,
	}, nil
}

func (manager *DatabaseManager) Setup() error {
	return manager.client.Setup()
}

func (manager *DatabaseManager) Close() error {
	return manager.client.Close()
}

func (manager *DatabaseManager) GetDBPath() string {
	return manager.client.GetDBPath()
}
