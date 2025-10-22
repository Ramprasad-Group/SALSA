# Extract Line Chart Data and Metadata

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
```bash
├── extern # Directory for holding external github repositories used here
    └── ChartDete # external github repo used to analye elements of a chart
    └── LineFormer # external github repo used to draw lines over the series on a line chart
├── src/
│ ├── input/ # Directory for placing input line chart images
│ ├── output/ # Directory where extracted data will be saved
│ ├── plextract/ # Main application logic for extraction
│ │ └── main.py # Entry point for running the extraction
    └── helpers for main.py
  ├── results/
│ ├── scripts/
│   └── setup.sh # Script to handle additional setup tasks
├── environment.yml # Conda environment definition
├── pyproject.toml
├── .gitmodules # Git submodule configuration
└── README.md # This README file
```

## Contributing

Contributions are welcome! If you find a bug or have an idea for an enhancement, please open an issue or submit a pull request.

## License

## Acknowledgements

-   Inspired by and built upon the work from [tdsone/extract-line-chart-data](https://github.com/tdsone/extract-line-chart-data).
-   Utilizes [LineFormer](link_to_lineformer_if_available) and [ChartDete](link_to_chartdete_if_available) for core functionalities.
-   This work was supported as part of the UNCAGE-ME, an Energy Frontier Research Center funded by the U.S. Department of Energy Office of Science, Basic Energy Sciences at the Georgia Institute of Technology under award # DE-SC0012577.