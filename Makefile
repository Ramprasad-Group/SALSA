.PHONY: setup env torch install ui clean help


CONDA_ENV   ?= AEM_OCR
SHELL       := /bin/bash
TORCH_VER ?= 1.13.1
TV_VER    ?= 0.14.1
CUDA_VER  ?= 11.7

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---- Full Setup (start here) -----------------------------------------

setup: env torch install ## One-command setup
	@echo ""
	@echo "===== Setup complete ====="
	@echo "Activate the env:   conda activate $(CONDA_ENV)"
	@echo "Launch the UI:      make ui"

env:
	conda env create -f environment.yml --name $(CONDA_ENV) || \
		conda env update -f environment.yml --name $(CONDA_ENV) --prune
	@echo "Activate with:  conda activate $(CONDA_ENV)"

torch:
	source "$$(conda info --base)/etc/profile.d/conda.sh" && \
	conda activate $(CONDA_ENV) && \
	conda config --env --set channel_priority strict && \
	conda remove -y pytorch torchvision torchaudio pytorch-cuda \
	  intel-openmp llvm-openmp mkl mkl-service libgomp || true && \
	conda install -y -c pytorch -c nvidia \
	  pytorch=$(TORCH_VER) torchvision=$(TV_VER) pytorch-cuda=$(CUDA_VER) \
	  intel-openmp mkl mkl-service && \
	python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

install:
	source "$$(conda info --base)/etc/profile.d/conda.sh" && \
	conda activate $(CONDA_ENV) && \
	bash src/setup-scripts/setup.sh
	
ui:
	source "$$(conda info --base)/etc/profile.d/conda.sh" && \
	conda activate $(CONDA_ENV) && \
	python -m streamlit run src/ui/streamlit_app.py

clean: ## Remove Python caches and egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/plextract.egg-info
