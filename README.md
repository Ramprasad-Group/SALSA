# SALSA - Semi-Autonomous Literature Scraping Assistant

SALSA (Semi-Autonomous Literature Scraping Assistant) is a pipeline for extracting numerical data and metadata from scientific literature, with specialized focus on line charts, materials science, and electrochemistry. Built on [LineFormer](https://github.com/williamschertzer/LineFormer) and [ChartDete](https://github.com/williamschertzer/ChartDete), extending the approach from [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data).

---

## Quick Start

Prerequisites: Conda or Miniconda, plus CUDA 11.7 for GPU acceleration. CPU-only runs are supported but slower.

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/Ramprasad-Group/SALSA
cd SALSA

# 2. Create/update the SALSA conda environment and install model dependencies
make setup
conda activate SALSA

# 3. Launch the UI
make ui
```

Manual setup is also supported:

```bash
conda env create -f environment.yml
conda activate SALSA
bash src/setup-scripts/setup.sh
streamlit run src/ui/streamlit_app.py
```

The setup script initializes submodules, checks Git LFS, pulls model artifacts, installs the MMDetection stack, patches bundled model configs, and installs SALSA in editable mode.

### Remote Hosting

Deployment templates for a remote Linux host are in [`deploy/`](./deploy/), including:

- [`deploy/DEPLOY_REMOTE.md`](./deploy/DEPLOY_REMOTE.md)
- [`deploy/systemd/salsa.service`](./deploy/systemd/salsa.service)
- [`deploy/nginx/salsa.conf`](./deploy/nginx/salsa.conf)
- [`deploy/env/salsa.env.example`](./deploy/env/salsa.env.example)

### LLM Configuration

SALSA uses an LLM for figure/table filtering, chemical composition extraction, and final dataset completion. The Streamlit UI can set the active model for interactive runs. For scripted or deployed runs, configure the same behavior with environment variables:

| Variable | Description | Example |
|---|---|---|
| `SALSA_LLM_PROVIDER` | General LLM provider: `ollama` or `openai` | `openai` |
| `SALSA_LLM_MODEL` | General LLM model | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API key, when using OpenAI | `sk-...` |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `OLLAMA_HOST` | Ollama endpoint, when using Ollama | `http://127.0.0.1:11434` |
| `SALSA_PROCESSING_LLM_PROVIDER` | Optional provider for the processing model | `openai` |
| `SALSA_PROCESSING_LLM_MODEL` | Optional separate model for dataset and chemistry processing | `gpt-4o` |
| `SALSA_PROCESSING_OPENAI_API_KEY` | Optional OpenAI key override for processing runs | `sk-...` |
| `SALSA_PROCESSING_OPENAI_BASE_URL` | Optional OpenAI-compatible base URL override for processing runs | `https://api.openai.com/v1` |

---

## How It Works

The primary workflow runs through the Streamlit UI and orchestrates the repo's pipeline modules:

1. **Paper acquisition and parsing** (`src/paper-downloader/`) downloads or reparses literature records, structured files, figures, tables, and full text.
2. **Relevance filtering** uses the configured LLM to decide which extracted figures and tables should move forward.
3. **Figure data extraction** (`src/extract_figure_data/`) digitizes selected line charts with ChartDete, LineFormer, OCR, legend matching, axis correction, and optional direct LLM figure extraction.
4. **Chemical composition extraction** (`src/chemical_composition/`) extracts and normalizes sample/composition evidence from text and tables, with optional JSME drawing support.
5. **Dataset completion** (`src/complete_dataset/`) combines figure data, tables, composition data, user-edited instructions, and LLM-generated transformation plans into final CSV outputs.

---

## Project Structure

```
SALSA/
├── extern/                          # Git submodules
│   ├── ChartDete/                   # Chart element detector (Cascade R-CNN + Swin-T)
│   └── LineFormer/                  # Line extraction transformer
├── src/
│   ├── ui/                          # Streamlit app and pipeline orchestration
│   │   └── streamlit_app.py
│   ├── paper-downloader/            # Paper acquisition + parsing
│   ├── extract_figure_data/         # Chart digitization and figure extraction
│   ├── chemical_composition/        # Chemistry/sample extraction helpers
│   ├── complete_dataset/            # LLM-assisted final dataset builder
│   ├── cost/                        # Usage/cost logging outputs
│   ├── ui_runs/                     # UI workspaces and run artifacts
│   ├── llm_client.py                # Unified OpenAI/Ollama client
│   └── setup-scripts/               # Setup and patch scripts
├── huggingface/                     # Model configs + checkpoints (Git LFS)
├── deploy/                          # Remote hosting templates
├── environment.yml                  # Conda environment
├── pyproject.toml                   # Python package metadata
├── Makefile                         # make setup / make ui
└── README.md
```

## Main Entry Points

| Command | Description |
|---|---|
| `make setup` | Create/update the `SALSA` conda env and install runtime/model dependencies |
| `make ui` | Launch the Streamlit UI inside the `SALSA` conda env |
| `streamlit run src/ui/streamlit_app.py` | Launch the UI manually from an activated env |
| `python src/extract_figure_data/plextract.py --help` | Run or inspect standalone chart digitization options |
| `python src/complete_dataset/complete_dataset.py --help` | Run or inspect standalone dataset completion options |
| `python src/paper-downloader/paper_downloader/download_records_multiple_dois_webofscience.py --help` | Inspect DOI/record acquisition options |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHARTDETE_DEVICE` | auto | ChartDete device, for example `cuda:0` or `cpu` |
| `LINEFORMER_DEVICE` | auto | LineFormer device, for example `cuda:0` or `cpu` |
| `CHARTDETE_HF_DIR` | `./huggingface` | ChartDete config/checkpoint directory |
| `LINEFORMER_HF_DIR` | `./huggingface` | LineFormer config/checkpoint directory |
| `CHARTDETE_CONF_THR` | `0.9` | ChartDete confidence threshold for crops |
| `SALSA_LLM_PROVIDER` | `ollama` | General LLM provider |
| `SALSA_LLM_MODEL` | required | General LLM model |
| `SALSA_PROCESSING_LLM_PROVIDER` | general provider | Optional provider override for dataset/composition processing |
| `SALSA_PROCESSING_LLM_MODEL` | general model | Optional model override for dataset/composition processing |
| `SALSA_PROCESSING_OPENAI_API_KEY` | `OPENAI_API_KEY` | Optional OpenAI key override for processing runs |
| `SALSA_PROCESSING_OPENAI_BASE_URL` | `OPENAI_BASE_URL` | Optional OpenAI-compatible base URL override for processing runs |
| `SALSA_CUSTOM_INSTRUCTIONS` | unset | Path to custom instruction JSON for a run |
| `SALSA_RUN_DIR` | run workspace | Directory used for run-scoped outputs and cost logs |
| `SALSA_RUN_ID` | generated | Run identifier used in UI/pipeline outputs |
| `SALSA_JSME_BIND_HOST` | `127.0.0.1` | Host for the optional JSME helper service |
| `SALSA_JSME_PORT` | `8765` | Port for the optional JSME helper service |
| `SALSA_JSME_PUBLIC_URL` | unset | Browser-reachable JSME URL for remote deployments |

---

## Contributing

Contributions are welcome! If you find a bug or have an idea for an enhancement, please open an issue or submit a pull request.

## Acknowledgements

- Built upon [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data)
- Uses [LineFormer](https://github.com/williamschertzer/LineFormer) and [ChartDete](https://github.com/williamschertzer/ChartDete)
- This work was supported as part of UNCAGE-ME, an Energy Frontier Research Center funded by the U.S. Department of Energy Office of Science, Basic Energy Sciences at the Georgia Institute of Technology under award # DE-SC0012577.
