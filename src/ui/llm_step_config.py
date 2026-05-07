from __future__ import annotations

LLM_STEP_EXTRACT_TABLES = "extract_tables"
LLM_STEP_CHEM_RELEVANT = "chem_relevant"
LLM_STEP_CHEM_COMPOSITION = "chem_composition"
LLM_STEP_FILTER_FIGURES = "filter_figures"
LLM_STEP_FILTER_TABLES = "filter_tables"
LLM_STEP_FIGURE_DATA = "figure_data"
LLM_STEP_COMPLETE_DATASET = "complete_dataset"

LLM_STEP_LABELS = {
    LLM_STEP_EXTRACT_TABLES: "Extract tables",
    LLM_STEP_CHEM_RELEVANT: "Chemical composition: find relevant content",
    LLM_STEP_CHEM_COMPOSITION: "Chemical composition: extract composition",
    LLM_STEP_FILTER_FIGURES: "Filter figures",
    LLM_STEP_FILTER_TABLES: "Filter tables",
    LLM_STEP_FIGURE_DATA: "Extract data from figures (LLM bypass)",
    LLM_STEP_COMPLETE_DATASET: "Complete dataset",
}

LLM_STEP_ENV_PREFIXES = {
    LLM_STEP_EXTRACT_TABLES: "AEM_OCR_STEP_EXTRACT_TABLES",
    LLM_STEP_CHEM_RELEVANT: "AEM_OCR_STEP_CHEM_RELEVANT",
    LLM_STEP_CHEM_COMPOSITION: "AEM_OCR_STEP_CHEM_COMPOSITION",
    LLM_STEP_FILTER_FIGURES: "AEM_OCR_STEP_FILTER_FIGURES",
    LLM_STEP_FILTER_TABLES: "AEM_OCR_STEP_FILTER_TABLES",
    LLM_STEP_FIGURE_DATA: "AEM_OCR_STEP_FIGURE_DATA",
    LLM_STEP_COMPLETE_DATASET: "AEM_OCR_STEP_COMPLETE_DATASET",
}

GENERAL_LLM_ENV_BY_STEP_SUFFIX = {
    "LLM_PROVIDER": "AEM_OCR_LLM_PROVIDER",
    "LLM_MODEL": "AEM_OCR_LLM_MODEL",
    "OPENAI_BASE_URL": "OPENAI_BASE_URL",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "OLLAMA_HOST": "OLLAMA_HOST",
    "OLLAMA_MODELS": "OLLAMA_MODELS",
    "OLLAMA_NUM_CTX": "AEM_OCR_OLLAMA_NUM_CTX",
}

GENERAL_LLM_ENV_VARS = tuple(GENERAL_LLM_ENV_BY_STEP_SUFFIX.values())


def step_env_prefix(step_key: str) -> str:
    return LLM_STEP_ENV_PREFIXES[step_key]


def step_env_name(step_key: str, suffix: str) -> str:
    return f"{step_env_prefix(step_key)}_{suffix}"


def step_env_names(step_key: str) -> tuple[str, ...]:
    return tuple(step_env_name(step_key, suffix) for suffix in GENERAL_LLM_ENV_BY_STEP_SUFFIX)


def step_env_to_general_env_map(step_key: str) -> dict[str, str]:
    prefix = step_env_prefix(step_key)
    return {
        f"{prefix}_{suffix}": general_env
        for suffix, general_env in GENERAL_LLM_ENV_BY_STEP_SUFFIX.items()
    }
