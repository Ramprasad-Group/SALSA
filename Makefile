.PHONY: setup env install ui clean help

CONDA_ENV   ?= AEM_OCR
SHELL       := /bin/bash

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---- Full Setup (start here) -----------------------------------------

setup: env install ## One-command setup: create conda env + install everything
	@echo ""
	@echo "===== Setup complete ====="
	@echo "Activate the env:   conda activate $(CONDA_ENV)"
	@echo "Launch the UI:      make ui"

env: ## Create / update the conda environment
	conda env create -f environment.yml --name $(CONDA_ENV) || \
		conda env update -f environment.yml --name $(CONDA_ENV) --prune
	@echo "Activate with:  conda activate $(CONDA_ENV)"

install: ## Install submodules, mmdet stack, patches (run inside active conda env)
	bash src/setup-scripts/setup.sh

# ---- Day-to-day -------------------------------------------------------

ui: ## Launch the Streamlit UI
	streamlit run src/ui/streamlit_app.py

clean: ## Remove Python caches and egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/plextract.egg-info
