package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// chatRequest is the JSON body expected by POST /chat.
type chatRequest struct {
	Prompt string  `json:"prompt"`
	MaxNew int     `json:"max_new"`
	Temp   float64 `json:"temp"`
	TopK   int     `json:"top_k"`
	TopP   float64 `json:"top_p"`
}

func (a *app) handleIndex() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		a.respond(w, "index.html", nil)
	}
}

func (a *app) handleChat() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("chat: invalid JSON body: %v", err)
			writeJSONError(w, http.StatusBadRequest, "invalid JSON body")
			return
		}

		if req.Prompt == "" {
			writeJSONError(w, http.StatusBadRequest, "prompt is empty")
			return
		}

		response, err := generateResponse(req)
		if err != nil {
			log.Printf("chat: generation error: %v", err)
			writeJSONError(w, http.StatusInternalServerError, "generation failed")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(map[string]string{"response": response}); err != nil {
			log.Printf("chat: error writing response: %v", err)
		}
	}
}

// repoRoot walks up from the running executable's directory until it finds a
// directory containing "config/web.json" (or "llm"), returning the repo root.
// It never hardcodes an absolute path.
func repoRoot() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("resolve executable: %w", err)
	}
	dir := filepath.Dir(exe)
	for {
		if _, err := os.Stat(filepath.Join(dir, "config", "web.json")); err == nil {
			return dir, nil
		}
		if _, err := os.Stat(filepath.Join(dir, "llm")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("could not locate repo root from %s", exe)
		}
		dir = parent
	}
}

// generateResponse runs the Python model bridge and returns the generated text.
func generateResponse(req chatRequest) (string, error) {
	root, err := repoRoot()
	if err != nil {
		return "", err
	}

	// Build the request JSON for the bridge. Only include fields that were
	// actually set so generate.py falls back to its config defaults otherwise.
	payload := map[string]any{}
	if req.Prompt != "" {
		payload["prompt"] = req.Prompt
	}
	if req.MaxNew != 0 {
		payload["max_new"] = req.MaxNew
	}
	if req.Temp != 0 {
		payload["temp"] = req.Temp
	}
	if req.TopK != 0 {
		payload["top_k"] = req.TopK
	}
	if req.TopP != 0 {
		payload["top_p"] = req.TopP
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	// Give the model a generous timeout so a hung process can't freeze the
	// server. Model load alone can take 10-30s on CPU.
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "python3", filepath.Join(root, "llm", "server_bridge.py"))
	cmd.Dir = root
	cmd.Stdin = bytes.NewReader(body)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return "", fmt.Errorf("model inference timed out after 120s")
		}
		return "", fmt.Errorf("run bridge: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	var result struct {
		Response string `json:"response"`
		Error    string `json:"error"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return "", fmt.Errorf("parse bridge output %q: %w", strings.TrimSpace(stdout.String()), err)
	}
	if result.Error != "" {
		return "", fmt.Errorf("model error: %s", result.Error)
	}
	return result.Response, nil
}

// writeJSONError writes a JSON error body with the given status code.
func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(map[string]string{"error": msg}); err != nil {
		log.Printf("error writing JSON error response: %v", err)
	}
}
