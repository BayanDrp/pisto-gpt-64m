# syntax=docker/dockerfile:1

# ---- Build the Go web server ----
FROM golang:1.22 AS gobuild
WORKDIR /src
COPY ui/ ./ui/
RUN cd ui && CGO_ENABLED=0 go build -o /out/pisto-server .

# ---- Runtime (Python + the Go server) ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Python deps for the model bridge (CPU torch; good for the 68M model).
# NOTE: serving the 792M AraGPT2 on CPU may exceed the 120s request timeout;
# use a GPU host / run locally for that model.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cli.py .
COPY config/ ./config/
# Align the served port (config/web.json) with EXPOSE / `make docker` (-p 8080:8080)
RUN python3 -c "import json,pathlib; p=pathlib.Path('config/web.json'); d=json.loads(p.read_text()); d['port']=8080; p.write_text(json.dumps(d, indent=2))"
COPY llm/ ./llm/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY training/ ./training/

# Compiled server from the build stage
COPY --from=gobuild /out/pisto-server /app/ui/pisto-server

# Mount your model weights here (weights/ from the host)
VOLUME ["/app/weights"]

EXPOSE 8080

CMD ["/app/ui/pisto-server"]
