// cmd/root.go
package cmd

import (
	"fmt"
	"github.com/joho/godotenv"
	"github.com/nlpfollower/deltamind/nexus/core"
	"github.com/spf13/cobra"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
)

var (
	port                 int
	persistentSessionURL string
	logDir               string
)

var rootCmd = &cobra.Command{
	Use:   "nexus",
	Short: "Nexus server for processing model requests",
	RunE:  runServer,
}

func init() {
	// Try to load .env from the project root first
	if err := godotenv.Load(filepath.Join("..", ".env")); err != nil {
		// If that fails, try current directory
		_ = godotenv.Load(".env")
	}

	rootCmd.Flags().IntVarP(&port, "port", "p", 8081, "Port to listen on")
	rootCmd.Flags().StringVarP(&persistentSessionURL, "session", "s", "", "URL for a persistent inference session (e.g. http://localhost:5000)")
	rootCmd.Flags().StringVar(&logDir, "log-dir", "", "Directory to save inference request logs (e.g. ./logs)")
}

func runServer(cmd *cobra.Command, args []string) error {
	// Create configuration
	cfg := &core.Config{
		Port:   port,
		LogDir: logDir,
	}

	// Create and start nexus
	nexus, err := core.NewNexus(cfg)
	if err != nil {
		return fmt.Errorf("failed to create nexus: %w", err)
	}

	if err := nexus.Start(); err != nil {
		return fmt.Errorf("failed to start nexus: %w", err)
	}

	log.Printf("Nexus server started on port %d", port)
	if logDir != "" {
		log.Printf("Request logging enabled to directory: %s", logDir)
	}

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Wait for shutdown signal
	<-sigChan
	log.Println("Shutting down...")

	// Graceful shutdown
	nexus.Stop()
	log.Println("Server stopped")

	return nil
}

func Execute() error {
	return rootCmd.Execute()
}
