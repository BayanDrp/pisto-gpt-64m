.PHONY: run server-build server-tunnel train-pretrain train-finetune docker

run:
	python cli.py

# Compile the Go web server into ui/pisto-server
server-build:
	go build -o ui/pisto-server ./ui/

# Run the server + expose it via a free Cloudflare tunnel (works on Colab)
server-tunnel:
	./scripts/run_server.sh

train-pretrain:
	python training/pretrain.py

train-finetune:
	python training/finetune.py

docker:
	docker build -t pisto-gpt .
	docker run -p 8080:8080 pisto-gpt
