<p align="center">
  <img src="./assets/salsa_icon.png" alt="SALSA logo" width="760"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-380/"><img src="https://img.shields.io/badge/python-3.8-blue.svg?logo=python&logoColor=white" alt="Python 3.8"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg?logo=streamlit&logoColor=white" alt="Streamlit UI"></a>
  <a href="./environment.yml"><img src="https://img.shields.io/badge/environment-conda-44a833.svg?logo=anaconda&logoColor=white" alt="Conda environment"></a>
  <a href="./Makefile"><img src="https://img.shields.io/badge/setup-make-064f8c.svg?logo=gnu&logoColor=white" alt="Make setup"></a>
  <a href="https://github.com/Ramprasad-Group/SALSA"><img src="https://img.shields.io/badge/GitHub-Ramprasad--Group%2FSALSA-181717.svg?logo=github&logoColor=white" alt="GitHub repository"></a>
</p>

# SALSA

**Semi-Autonomous Literature Scraping Assistant for extracting structured data from scientific papers, figures, tables, and uploaded images.**

## Overview

SALSA is a Streamlit-driven literature extraction pipeline for materials science workflows. It downloads or parses scientific papers, filters relevant figures and tables, digitizes line charts, extracts chemical composition evidence, and combines intermediate outputs into completed datasets.

The pipeline builds on [LineFormer](https://github.com/williamschertzer/LineFormer) and [ChartDete](https://github.com/williamschertzer/ChartDete), extending the approach from [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data).

**Key Features:**

- Literature ingestion from DOI lists, local PDFs, or standalone figure images
- Streamlit UI for workspaces, settings presets, run history, review, and interventions
- Chart digitization with ChartDete, LineFormer, OCR, legend matching, and axis correction
- LLM-assisted figure/table filtering, table extraction, chemistry extraction, and dataset completion
- Human-in-the-loop tools for crop selection, axis calibration, mask overrides, and reruns
- Per-run usage and cost logging for model-backed extraction steps
- Optional remote deployment templates for systemd and Nginx

## Installation

Prerequisites:

- Conda or Miniconda
- Git submodule support
- CUDA 11.7 for GPU acceleration, or CPU-only execution for slower local runs
- LLM provider credentials or an Ollama server for model-backed steps

Install from source:

```bash
git clone --recurse-submodules https://github.com/Ramprasad-Group/SALSA.git
cd SALSA
make setup
conda activate SALSA
```

Manual setup is also supported:

```bash
conda env create -f environment.yml
conda activate SALSA
bash src/setup-scripts/setup.sh
```

The setup script initializes submodules, checks Git LFS, pulls model artifacts, installs the MMDetection stack, patches bundled model configs, and installs SALSA in editable mode.

## Quick Start

Launch the UI:

```bash
make ui
```

Or run Streamlit directly from an activated environment:

```bash
streamlit run src/ui/streamlit_app.py
```

Then start a run from the sidebar:

1. Choose or create a workspace.
2. Select **DOI(s)**, **PDF(s)**, or **Figure(s)** as the input source.
3. Paste DOI values or upload files.
4. Choose settings presets and LLM models.
5. Click **Run pipeline**.
6. Review extracted tables, chart data, composition evidence, completed datasets, and intervention tools.

## Workflow

SALSA's main workflow is orchestrated by the Streamlit UI:

1. **Paper acquisition and parsing** downloads DOI records or parses uploaded PDFs into full text, tables, figures, and metadata.
2. **Relevance filtering** uses the configured LLM to select figures and tables worth extracting.
3. **Figure data extraction** digitizes selected line charts with detection, OCR, legend matching, and optional direct LLM figure extraction.
4. **Chemical composition extraction** identifies sample/composition evidence from paper text, tables, and user-drawn structures.
5. **Dataset completion** combines figures, tables, compositions, custom instructions, and LLM-generated transformation plans into final CSV outputs.
6. **Human review and interventions** let users crop figures, calibrate axes, edit series mappings, apply masks, and rerun extraction stages.

## Core Capabilities

### Input Sources

- DOI lists for paper download and parsing
- Local PDF uploads
- Standalone figure image uploads

### Extraction Targets

- Line chart data
- Figure and table metadata
- Structured table rows
- Chemical composition and sample evidence
- Completed per-paper and per-run datasets

### Review Tools

- Workspace and run browser
- Settings and instruction presets
- Axis title and axis calibration overrides
- Crop-based subfigure extraction
- OCR and LLM reruns
- Cost and usage summaries

## LLM Configuration

SALSA uses LLMs for figure/table filtering, chemical composition extraction, table extraction, figure extraction, and final dataset completion. The UI can set model choices for interactive runs. Scripted or deployed runs can use environment variables:

| Variable | Description | Example |
|---|---|---|
| `SALSA_LLM_PROVIDER` | General LLM provider: `ollama` or `openai` | `openai` |
| `SALSA_LLM_MODEL` | General LLM model | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API key, when using OpenAI | `sk-...` |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `OLLAMA_HOST` | Ollama endpoint, when using Ollama | `http://127.0.0.1:11434` |
| `SALSA_OLLAMA_NUM_CTX` | Optional shared Ollama context window | `16384` |
| `SALSA_PROCESSING_LLM_PROVIDER` | Optional provider for dataset and chemistry processing | `openai` |
| `SALSA_PROCESSING_LLM_MODEL` | Optional separate processing model | `gpt-4o` |
| `SALSA_PROCESSING_OPENAI_API_KEY` | Optional OpenAI key override for processing runs | `sk-...` |
| `SALSA_PROCESSING_OPENAI_BASE_URL` | Optional OpenAI-compatible base URL override | `https://api.openai.com/v1` |
| `SALSA_PROCESSING_OLLAMA_NUM_CTX` | Optional Ollama context window override | `32768` |

## Project Structure

```text
SALSA/
├── assets/                          # README and UI assets
├── deploy/                          # Remote hosting templates
│   ├── env/
│   ├── nginx/
│   └── systemd/
├── extern/                          # Git submodules
│   ├── ChartDete/                   # Chart element detector
│   └── LineFormer/                  # Line extraction transformer
├── huggingface/                     # Model configs and checkpoints
├── src/
│   ├── ui/                          # Streamlit app and orchestration
│   ├── paper-downloader/            # Paper acquisition and parsing
│   ├── extract_figure_data/         # Chart digitization and figure extraction
│   ├── chemical_composition/        # Chemistry/sample extraction helpers
│   ├── complete_dataset/            # LLM-assisted final dataset builder
│   ├── cost/                        # Usage and cost summaries
│   ├── setup-scripts/               # Setup and patch scripts
│   └── ui_runs/                     # Local UI workspaces and run artifacts
├── tests/                           # Focused regression tests
├── environment.yml                  # Conda environment
├── Makefile                         # make setup / make ui
├── pyproject.toml                   # Python package metadata
└── README.md
```

## Main Entry Points

| Command | Description |
|---|---|
| `make setup` | Create or update the `SALSA` conda environment and install runtime/model dependencies |
| `make ui` | Launch the Streamlit UI inside the `SALSA` conda environment |
| `streamlit run src/ui/streamlit_app.py` | Launch the UI manually from an activated environment |
| `python src/extract_figure_data/plextract.py --help` | Inspect standalone chart digitization options |
| `python src/complete_dataset/complete_dataset.py --help` | Inspect standalone dataset completion options |
| `python src/paper-downloader/paper_downloader/download_records_multiple_dois_webofscience.py --help` | Inspect DOI/record acquisition options |

## Deployment

Remote hosting templates are available in [`deploy/`](./deploy/):

- [`deploy/DEPLOY_REMOTE.md`](./deploy/DEPLOY_REMOTE.md)
- [`deploy/systemd/salsa.service`](./deploy/systemd/salsa.service)
- [`deploy/nginx/salsa.conf`](./deploy/nginx/salsa.conf)
- [`deploy/env/salsa.env.example`](./deploy/env/salsa.env.example)

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
| `SALSA_PROCESSING_OPENAI_BASE_URL` | `OPENAI_BASE_URL` | Optional OpenAI-compatible base URL override |
| `SALSA_CUSTOM_INSTRUCTIONS` | unset | Path to custom instruction JSON for a run |
| `SALSA_RUN_DIR` | run workspace | Directory used for run-scoped outputs and cost logs |
| `SALSA_RUN_ID` | generated | Run identifier used in UI/pipeline outputs |
| `SALSA_JSME_BIND_HOST` | `127.0.0.1` | Host for the optional JSME helper service |
| `SALSA_JSME_PORT` | `8765` | Port for the optional JSME helper service |
| `SALSA_JSME_PUBLIC_URL` | unset | Browser-reachable JSME URL for remote deployments |

## Acknowledgements

- Built upon [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data)
- Uses [LineFormer](https://github.com/williamschertzer/LineFormer) and [ChartDete](https://github.com/williamschertzer/ChartDete)
- Supported as part of UNCAGE-ME, an Energy Frontier Research Center funded by the U.S. Department of Energy Office of Science, Basic Energy Sciences at the Georgia Institute of Technology under award # DE-SC0012577.
