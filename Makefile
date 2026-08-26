.PHONY: help install train-pretrain train-finetune train-finetune-legacy app docker clean

help:
	@echo "Pisto GPT Makefile"
	@echo "  install                 Install Python deps"
	@echo "  train-pretrain          Pretrain the 68M model from scratch"
	@echo "  train-finetune          Fine-tune AraGPT2 (HuggingFace)  [primary]"
	@echo "  train-finetune-legacy   Fine-tune the legacy 68M model"
	@echo "  app                     Launch the Go web server + model bridge"
	@echo "  docker                  Build & run the app in Docker (port 8080)"
	@echo "  clean                   Remove checkpoints/__pycache__"

install:
	python -m pip install -r requirements.txt

train-pretrain:
	python training/pretrain.py

train-finetune:
	python training/finetune_hf.py

train-finetune-legacy:
	python training/finetune.py

app:
	go build -o ui/pisto-server ./ui && ./ui/pisto-server

docker:
	docker build -t pisto-gpt .
	docker run -p 8080:8080 -v "$(CURDIR)/weights:/app/weights" pisto-gpt

clean:
	rm -rf checkpoints __pycache__ training/__pycache__ llm/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
