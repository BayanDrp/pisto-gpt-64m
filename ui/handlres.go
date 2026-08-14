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
	"strconv"
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

// configRequest is the JSON body expected by POST /config. Pointers are used
// so missing fields can be distinguished from zero values; all four generation
// fields are required, rep_pen is optional.
type configRequest struct {
	MaxNew *int     `json:"max_new"`
	Temp   *float64 `json:"temp"`
	TopK   *int     `json:"top_k"`
	TopP   *float64 `json:"top_p"`
	RepPen *float64 `json:"rep_pen"`
}

// keepZeroFloat is a float64 that always marshals with exactly one decimal
// place, so a value of 0.0 round-trips as "0.0" instead of "0" (the default
// Go encoding for a zero float64). Used for "dropout" in the model block.
type keepZeroFloat float64

func (f keepZeroFloat) MarshalJSON() ([]byte, error) {
	return []byte(strconv.FormatFloat(float64(f), 'f', 1, 64)), nil
}

// generateModel mirrors the "model" object in config/generate.json. The field
// order matches the original file so json.Marshal preserves it byte-for-byte.
type generateModel struct {
	DModel            int           `json:"d_model"`
	NHead             int           `json:"nhead"`
	DimFeedforward    int           `json:"dim_feedforward"`
	Dropout           keepZeroFloat `json:"dropout"`
	TransformerLayers int           `json:"transformer_layers"`
	MaxLen            int           `json:"max_len"`
}

// generationSettings mirrors the "generation" object in config/generate.json.
// Field order matches the original file.
type generationSettings struct {
	MaxNew int     `json:"max_new"`
	Temp   float64 `json:"temp"`
	TopK   int     `json:"top_k"`
	TopP   float64 `json:"top_p"`
	RepPen float64 `json:"rep_pen"`
}

// generateConfig mirrors config/generate.json. The field order matches the
// original file (model, weight_path, generation) so json.Marshal preserves it,
// avoiding the key re-ordering that map[string]any would produce.
type generateConfig struct {
	Model      generateModel      `json:"model"`
	WeightPath string             `json:"weight_path"`
	Generation generationSettings `json:"generation"`
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

// configPath returns the absolute path to config/generate.json at the repo root.
func configPath() (string, error) {
	root, err := repoRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "config", "generate.json"), nil
}

// handleGetConfig returns the current generation settings from
// config/generate.json as JSON, e.g. {"max_new":100,"temp":0.5,...}.
func (a *app) handleGetConfig() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		path, err := configPath()
		if err != nil {
			log.Printf("config: resolve path: %v", err)
			writeJSONError(w, http.StatusInternalServerError, "could not locate config/generate.json")
			return
		}

		data, err := os.ReadFile(path)
		if err != nil {
			log.Printf("config: read %s: %v", path, err)
			writeJSONError(w, http.StatusInternalServerError, "could not read config/generate.json")
			return
		}

		var cfg generateConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			log.Printf("config: parse %s: %v", path, err)
			writeJSONError(w, http.StatusInternalServerError, "could not parse config/generate.json")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(cfg.Generation); err != nil {
			log.Printf("config: error writing response: %v", err)
		}
	}
}

// handleSaveConfig updates the "generation" keys of config/generate.json while
// leaving "model" and "weight_path" untouched. It unmarshals into a typed
// struct (not map[string]any) so json.MarshalIndent preserves the original
// field order and the "dropout": 0.0 formatting.
func (a *app) handleSaveConfig() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var req configRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("config: invalid JSON body: %v", err)
			writeJSONError(w, http.StatusBadRequest, "invalid JSON body")
			return
		}

		if req.MaxNew == nil || req.Temp == nil || req.TopK == nil || req.TopP == nil {
			writeJSONError(w, http.StatusBadRequest, "missing required field (max_new, temp, top_k, top_p)")
			return
		}

		path, err := configPath()
		if err != nil {
			log.Printf("config: resolve path: %v", err)
			writeJSONError(w, http.StatusInternalServerError, "could not locate config/generate.json")
			return
		}

		data, err := os.ReadFile(path)
		if err != nil {
			log.Printf("config: read %s: %v", path, err)
			writeJSONError(w, http.StatusInternalServerError, "could not read config/generate.json")
			return
		}

		var cfg generateConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			log.Printf("config: parse %s: %v", path, err)
			writeJSONError(w, http.StatusInternalServerError, "could not parse config/generate.json")
			return
		}

		cfg.Generation.MaxNew = *req.MaxNew
		cfg.Generation.Temp = *req.Temp
		cfg.Generation.TopK = *req.TopK
		cfg.Generation.TopP = *req.TopP
		if req.RepPen != nil {
			cfg.Generation.RepPen = *req.RepPen
		}

		out, err := json.MarshalIndent(cfg, "", "    ")
		if err != nil {
			log.Printf("config: marshal %s: %v", path, err)
			writeJSONError(w, http.StatusInternalServerError, "could not serialize config/generate.json")
			return
		}
		out = append(out, '\n')

		if err := os.WriteFile(path, out, 0644); err != nil {
			log.Printf("config: write %s: %v", path, err)
			writeJSONError(w, http.StatusInternalServerError, "could not write config/generate.json")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if err := json.NewEncoder(w).Encode(map[string]bool{"ok": true}); err != nil {
			log.Printf("config: error writing response: %v", err)
		}
	}
}

// handleConfigMethodNotAllowed is the fallback for /config requests whose
// method is neither GET nor POST. The method-specific patterns above handle
// GET and POST; this catches everything else (PUT, DELETE, PATCH, ...) and
// rejects it with 405. Without it, the catch-all "/" route would swallow
// these requests and return 404 instead.
func (a *app) handleConfigMethodNotAllowed() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Allow", "GET, POST")
		writeJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
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
