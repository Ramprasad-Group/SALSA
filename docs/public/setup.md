SALSA can be used through the hosted web application or installed from source.
You do not need a local installation when an administrator has provided a hosted
SALSA address.

## Use the hosted application

1. Select **Sign in / Open SALSA** in the documentation sidebar.
2. Sign in, or submit an account request if account creation is enabled.
3. After an administrator approves the account, sign in and create a workspace.
4. Open the **User guide** for instructions on configuring and starting a run.

Accounts and workspaces are private. The public documentation does not expose
run data, settings, API credentials, or administration controls.

## Install SALSA from source

### Prerequisites

- Conda or Miniconda
- Git with submodule support
- CUDA 11.7 for GPU acceleration, or CPU execution for slower local runs
- Credentials for the model and literature providers you intend to use, or an
  accessible Ollama server

Clone the repository and run the supported setup target:

```bash
git clone --recurse-submodules https://github.com/Ramprasad-Group/SALSA.git
cd SALSA
make setup
conda activate SALSA
```

The manual equivalent is:

```bash
conda env create -f environment.yml
conda activate SALSA
bash src/setup-scripts/setup.sh
```

The setup script initializes the Git submodules, checks Git LFS, retrieves model
artifacts, installs the MMDetection stack, applies the bundled model patches,
and installs SALSA in editable mode.

## Start the local application

From the repository root with the `SALSA` environment active, run:

```bash
make ui
```

Alternatively:

```bash
streamlit run src/ui/streamlit_app.py
```

Open the address printed by Streamlit in a browser. Use the public guide and
setup pages without signing in; an account is still required to access
workspaces or run the extraction pipeline.

## Provider configuration

Configure only the providers needed by your workflow. Provider credentials
should be supplied through the deployment environment or saved through SALSA's
authenticated settings interface. Never place API keys in public documentation
or commit them to the repository.

- Model-backed stages can use OpenAI-compatible models or Ollama.
- DOI downloads may require publisher-specific text-and-data-mining credentials
  and appropriate institutional access.

For server deployment, use the templates and operator documentation in the
repository's `deploy/` directory. Those instructions are intended for the
administrator operating the SALSA host.

## Responsible use

Only process documents and data for which you have the necessary rights,
permissions, or lawful basis. SALSA does not grant access rights to publisher
content and must not be used to bypass access controls or license restrictions.
