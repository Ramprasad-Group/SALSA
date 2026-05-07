# SALSA — Semi-Autonomous Literature Scraping Assistant

SALSA (Semi-Autonomous Literature Scraping Assistant) is a pipeline for extracting numerical data and metadata from scientific literature, with specialized focus on line charts, materials science, and electrochemistry. Built on [LineFormer](https://github.com/williamschertzer/LineFormer) and [ChartDete](https://github.com/williamschertzer/ChartDete), extending the approach from [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data).

---

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/Ramprasad-Group/SALSA
cd SALSA

# 2. Create conda environment
conda env create -f environment.yml
conda activate AEM_OCR

# 3. Run setup (installs submodules, mmdet stack, patches)
bash src/setup-scripts/setup.sh

# 4. Launch the UI
streamlit run src/ui/streamlit_app.py
```

Or use `make`:

```bash
make setup          # Steps 2 + 3
conda activate AEM_OCR
make ui             # Step 4
```

### Remote Hosting

Deployment templates for a remote Linux host are in [`deploy/`](./deploy/), including:

- [`deploy/DEPLOY_REMOTE.md`](./deploy/DEPLOY_REMOTE.md)
- [`deploy/systemd/aem-ocr.service`](./deploy/systemd/aem-ocr.service)
- [`deploy/nginx/aem-ocr.conf`](./deploy/nginx/aem-ocr.conf)
- [`deploy/env/aem-ocr.env.example`](./deploy/env/aem-ocr.env.example)

### Prerequisites

- **Git LFS** — model checkpoints are stored with LFS. Install it if not already:
  ```bash
  git lfs install
  ```
- **Conda** (Miniconda or Anaconda)
- **CUDA 11.7** (optional, for GPU acceleration). CPU-only works but is slower.

### LLM Configuration

The dataset-completion and figure-filtering stages require an LLM. Configure via environment variables before launching the UI:

| Variable | Description | Example |
|---|---|---|
| `AEM_OCR_LLM_PROVIDER` | `ollama` or `openai` | `openai` |
| `AEM_OCR_LLM_MODEL` | Model name | `gpt-4o` |
| `OPENAI_API_KEY` | API key (OpenAI only) | `sk-...` |
| `OLLAMA_HOST` | Ollama endpoint (Ollama only) | `http://127.0.0.1:11434` |

---

## How It Works

The pipeline has three stages:

1. **Paper Acquisition** (`src/paper-downloader/`) — Download papers by DOI, extract figures and tables from structured files (XML/HTML).

2. **Figure Data Extraction** (`src/extract_figure_data/plextract.py`) — For each figure:
   - **ChartDete**: detect chart elements (axes, legends, plot areas)
   - **LineFormer**: extract line series as polylines
   - **TrOCR**: OCR axis labels, titles, and legend text
   - **Legend matching**: pair legend entries to line series (Hungarian algorithm)
   - **Coordinate correction**: transform pixel coordinates to data space
   - Export labeled (x, y) datasets

3. **Dataset Completion** (`src/complete_dataset/complete_dataset.py`) — Merge extracted figure data with table data, use an LLM to map columns to a target output schema, produce final CSVs.

---

## Project Structure

```
SALSA/
├── extern/                          # Git submodules
│   ├── ChartDete/                   # Chart element detector (Cascade R-CNN + Swin-T)
│   └── LineFormer/                  # Line extraction transformer
├── src/
│   ├── extract_figure_data/         # Core extraction pipeline
│   │   ├── plextract.py             # Main orchestrator
│   │   ├── run_chartdete.py         # ChartDete wrapper
│   │   ├── run_lineformer.py        # LineFormer wrapper
│   │   ├── run_ocr.py              # TrOCR wrapper
│   │   ├── match_legend.py          # Legend-to-series matching
│   │   ├── correct_coordinates.py   # Pixel → data coordinate transform
│   │   └── finalize_labels.py       # Export datasets + plots
│   ├── complete_dataset/            # LLM-driven dataset completion
│   │   └── complete_dataset.py
│   ├── ui/                          # Streamlit interactive UI
│   │   └── streamlit_app.py
│   ├── paper-downloader/            # Paper acquisition + parsing
│   ├── llm_client.py               # Unified OpenAI / Ollama client
│   └── setup-scripts/              # Setup and patch scripts
│       ├── setup.sh
│       ├── patch_config.py
│       └── patch_lineformer.py
├── huggingface/                     # Model configs + checkpoints (Git LFS)
├── environment.yml                  # Conda environment
├── pyproject.toml                   # Package metadata (plextract)
├── Makefile                         # make setup / make ui
└── README.md
```

## Main Entry Points

| Command | Description |
|---|---|
| `streamlit run src/ui/streamlit_app.py` | Interactive web UI (recommended) |
| `python src/extract_figure_data/plextract.py --input <dir>` | Run figure extraction standalone |
| `python src/complete_dataset/complete_dataset.py` | Run dataset completion standalone |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHARTDETE_DEVICE` | auto (cuda/cpu) | Device for ChartDete inference |
| `LINEFORMER_DEVICE` | auto (cuda/cpu) | Device for LineFormer inference |
| `CHARTDETE_HF_DIR` | `./huggingface` | Directory for ChartDete model files |
| `LINEFORMER_HF_DIR` | `./huggingface` | Directory for LineFormer model files |
| `AEM_OCR_LLM_PROVIDER` | `ollama` | LLM provider (`ollama` or `openai`) |
| `AEM_OCR_LLM_MODEL` | — | Model name (required for LLM features) |

SALSA currently keeps the legacy `AEM_OCR_*` environment variable names for backwards compatibility with existing deployments and scripts.

---

## Contributing

Contributions are welcome! If you find a bug or have an idea for an enhancement, please open an issue or submit a pull request.

## Acknowledgements

- Built upon [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data)
- Uses [LineFormer](https://github.com/williamschertzer/LineFormer) and [ChartDete](https://github.com/williamschertzer/ChartDete)
- This work was supported as part of UNCAGE-ME, an Energy Frontier Research Center funded by the U.S. Department of Energy Office of Science, Basic Energy Sciences at the Georgia Institute of Technology under award # DE-SC0012577.
