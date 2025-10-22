# Extract Line Chart Data

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Submodules](https://img.shields.io/badge/submodules-included-brightgreen.svg)](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

A wrapper for LineFormer and ChartDete, offering an improved approach over the original code from [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data) for extracting data from line charts.

---

## Table of Contents

- [Setup](#setup)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Setup

Follow these steps to set up your environment and dependencies:

1.  **Clone the Repository with Submodules:**
    Ensure you clone the repository including its submodules, as this project relies on external tools.
    ```bash
    git clone --recurse-submodules https://github.com/Ramprasad-Group/AEM_OCR
    ```
2.  **Create and Activate Conda Environment:**
    Navigate into the cloned repository and create a new Conda environment using the provided `environment.yml` file.
    ```bash
    cd AEM_OCR # Or the name of your cloned directory if different
    conda env create -f environment.yml
    conda activate AEM_OCR # Assuming the environment name is AEM_OCR from the yml file
    ```
3.  **Make Setup Script Executable:**
    Grant execution permissions to the setup script.
    ```bash
    chmod +x src/scripts/setup.sh
    ```
4.  **Run Setup Script:**
    Execute the setup bash script to finalize the environment configuration.
    ```bash
    bash src/scripts/setup.sh
    ```

## How to Use

Once the setup is complete, you can use the tool as follows:

1.  **Place Figures:**
    Put all the line chart images you wish to process into the `src/input` directory.
2.  **Run the Extraction Script:**
    Execute the main Python script to start the data extraction process.
    ```bash
    python src/plextract/main.py
    ```
    The extracted data will be saved in an output directory, typically `src/output`.

## Project Structure