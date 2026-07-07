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

## Project Structure

```text
SALSA/
├── assets                          # UI images
├── deploy                          # Remote deployment documentation and templates
│   ├── env
│   ├── nginx
│   └── systemd
├── extern                          # Git submodules used for chart digitization
│   ├── ChartDete
│   └── LineForme/
├── huggingface                     # ChartDete and LineFormer configs/checkpoints
├── src
│   ├── chemical_composition        # Composition extraction and JSME drawing tool
│   ├── complete_dataset            # Dataset assembly and row transformation
│   ├── cost                        # Usage and cost summaries
│   ├── extract_figure_data         # OCR, chart detection, masks, and XY extraction
│   ├── paper-downloader            # Paper acquisition and parsing
│   ├── setup-scripts               # Environment setup and submodule patching
│   ├── ui                          # Streamlit UI, pipeline, auth, and interventions
│   ├── ui_runs                     # Generated workspaces and run artifacts (ignored)
│   └── llm_client.py               # Shared LLM provider client
├── tests                           # Pipeline, UI, schema, prompt, and cost tests
├── .gitmodules                     # ChartDete and LineFormer submodule definitions
├── environment.yml                 # Conda environment
├── Makefile                        # Setup, launch, and cleanup commands
├── pyproject.toml                  # Python package metadata
└── README.md
```

Runtime logs, extraction results, workspaces, Python caches, and generated
`*.egg-info` directories are not part of the source layout and are ignored.

## Deployment

Remote hosting templates are available in [`deploy/`](./deploy/):

- [`deploy/DEPLOY_REMOTE.md`](./deploy/DEPLOY_REMOTE.md)
- [`deploy/systemd/salsa.service`](./deploy/systemd/salsa.service)
- [`deploy/systemd/salsa-dev.service`](./deploy/systemd/salsa-dev.service)
- [`deploy/nginx/salsa.conf`](./deploy/nginx/salsa.conf)
- [`deploy/env/salsa.env.example`](./deploy/env/salsa.env.example)
- [`deploy/env/salsa-dev.env.example`](./deploy/env/salsa-dev.env.example)

Production disables Streamlit source-file watching so code edits do not reload
the production app while runs are active. Use `salsa-dev.service` on a separate
port and development data root to test changes, then restart production
deliberately after validation.

## Acknowledgements

- Built upon [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data)
- Uses [LineFormer](https://github.com/williamschertzer/LineFormer) and [ChartDete](https://github.com/williamschertzer/ChartDete)
- Supported as part of UNCAGE-ME, an Energy Frontier Research Center funded by the U.S. Department of Energy Office of Science, Basic Energy Sciences at the Georgia Institute of Technology under award # DE-SC0012577.


## Responsible Use and Publisher Compliance

- SALSA is provided as a research software tool for assisting users with literature data extraction, document parsing, table extraction, figure digitization, and dataset curation. Users are solely responsible for ensuring that their use of SALSA complies with all applicable laws, publisher agreements, institutional policies, database licenses, API terms, copyright restrictions, text-and-data-mining policies, and model-provider terms.

- SALSA does not grant users any rights to access, download, reproduce, redistribute, mine, extract, store, or publish third-party documents, figures, tables, text, metadata, or datasets. Users must only process documents and data for which they have the necessary rights, permissions, or lawful basis. SALSA must not be used to circumvent paywalls, access controls, robots.txt restrictions, publisher APIs, license limitations, or other technical or contractual restrictions.

- The developers, contributors, and affiliated institutions are not responsible for any user’s violation of publisher agreements, copyright law, database rights, institutional access policies, data-use restrictions, or other third-party terms. Users are responsible for reviewing, verifying, and validating all extracted outputs before use, redistribution, publication, or downstream analysis.
