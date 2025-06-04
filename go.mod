module github.com/nlpfollower/deltamind/nexus

go 1.23.1

replace github.com/nlpfollower/deltamind/database => ./../database

replace github.com/nlpfollower/deltamind/orchestration => ./../orchestration

require (
	github.com/google/uuid v1.6.0
	github.com/joho/godotenv v1.5.1
	github.com/nlpfollower/deltamind/database v0.0.0-00010101000000-000000000000
	github.com/nlpfollower/deltamind/orchestration v0.0.0-00010101000000-000000000000
	github.com/sashabaranov/go-openai v1.32.3
	github.com/spf13/cobra v1.8.1
	github.com/stretchr/testify v1.9.0
)

require (
	github.com/boltdb/bolt v1.3.1 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/spf13/pflag v1.0.5 // indirect
	golang.org/x/sys v0.26.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
