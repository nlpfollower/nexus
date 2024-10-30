package main

import (
	"log"

	"github.com/nlpfollower/deltamind/nexus/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		log.Fatalf("Error executing nexus server: %v", err)
	}
}
