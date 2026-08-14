# Stage 1: build the Go web server
FROM golang:1.23 AS builder

WORKDIR /src

# The Go server uses only the standard library, so we can copy the whole
# module and build straight away.
COPY go.mod ./
COPY ui/ ./ui/

RUN go build -o /out/pisto-server ./ui/

# Stage 2: Python runtime + compiled Go server
FROM python:3.11-slim

WORKDIR /app

# Python deps (torch CPU build via the cpu index)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Repo contents (config/, llm/, training/, ui/, scripts/, cli.py, requirements.txt)
COPY config/ ./config/
COPY llm/ ./llm/
COPY training/ ./training/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY cli.py ./cli.py
COPY requirements.txt ./requirements.txt

# Compiled Go server from the builder stage
COPY --from=builder /out/pisto-server ./ui/pisto-server

EXPOSE 8080

# The Go server runs with CWD=/app so it finds ui/templates, ui/static and
# config/web.json; the Python bridge is spawned as `python3 llm/server_bridge.py`
# with cmd.Dir = /app, so python3 must be on PATH (it is, from python:3.11-slim).
CMD ["./ui/pisto-server"]