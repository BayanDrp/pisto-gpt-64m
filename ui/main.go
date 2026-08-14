package main

import (
	"encoding/json"
	"html/template"
	"log"
	"net/http"
	"os"
	"strconv"
)

type app struct {
	template *template.Template
	mux      *http.ServeMux
}

func newApp() *app {
	a := &app{
		template: loadTemplates(),
		mux:      http.NewServeMux(),
	}
	a.routes()
	return a
}

// loadTemplates parses the HTML templates. It first tries the repo-root
// relative path (ui/templates/*.html), then falls back to templates/*.html
// for when the binary is run from inside the ui/ directory.
func loadTemplates() *template.Template {
	patterns := []string{"ui/templates/*.html", "templates/*.html"}
	for _, pattern := range patterns {
		tmpl, err := template.ParseGlob(pattern)
		if err == nil && tmpl != nil {
			return tmpl
		}
		log.Printf("template glob %q failed: %v", pattern, err)
	}
	log.Fatal("no templates found (tried ui/templates/*.html and templates/*.html)")
	return nil
}

// staticDir returns the directory serving static assets, preferring the
// repo-root relative path and falling back to the ui/ directory.
func staticDir() string {
	if _, err := os.Stat("ui/static"); err == nil {
		return "ui/static"
	}
	return "static"
}

func (a *app) routes() {
	a.mux.HandleFunc("/", a.handleIndex())
	a.mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir()))))
	a.mux.HandleFunc("/chat", a.handleChat())
}

func (a *app) respond(w http.ResponseWriter, name string, data any) {
	if err := a.template.ExecuteTemplate(w, name, data); err != nil {
		log.Printf("template execution error: %v", err)
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}
}

func main() {
	port := loadPort()

	app := newApp()

	addr := ":" + port
	log.Printf("Pisto GPT server listening on %s", addr)
	if err := http.ListenAndServe(addr, app.mux); err != nil {
		log.Fatal(err)
	}
}

// loadPort reads the port from config/web.json at the repo root. It falls
// back to 8080 if the file is missing, unreadable, or has no port set.
func loadPort() string {
	file, err := os.Open("config/web.json")
	if err != nil {
		log.Printf("warning: could not open config/web.json: %v; using default port 8080", err)
		return "8080"
	}
	defer file.Close()

	var config struct {
		Port int `json:"port"`
	}
	if err := json.NewDecoder(file).Decode(&config); err != nil {
		log.Printf("warning: could not decode config/web.json: %v; using default port 8080", err)
		return "8080"
	}
	if config.Port == 0 {
		log.Printf("warning: no port set in config/web.json; using default port 8080")
		return "8080"
	}
	return strconv.Itoa(config.Port)
}