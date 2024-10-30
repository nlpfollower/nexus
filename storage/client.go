package storage

import (
	"github.com/nlpfollower/deltamind/database/db"
)

type DatabaseClient struct {
	*db.ProtectedDatabase
	dbPath string
}

func NewDatabaseClient(dbPath string) *DatabaseClient {
	boltDB := db.NewBoltDatabase(dbPath)
	protectedDB := db.NewProtectedDatabase(boltDB)
	return &DatabaseClient{
		ProtectedDatabase: protectedDB,
		dbPath:            dbPath,
	}
}

func (client *DatabaseClient) GetDBPath() string {
	return client.dbPath
}
