# SuperArchitect

## Overview

SuperArchitect is a Python-based command-line tool designed to assist with architectural planning by leveraging multiple AI models. It takes a high-level architectural request, processes it through a structured workflow involving decomposition, multi-model consultation, analysis, and synthesis, ultimately generating a proposed architectural plan or step-by-step guide.

## Features

*   **Multi-AI Consultation:** Utilizes various AI models (configurable: OpenAI, Anthropic Claude, Google Gemini) for diverse perspectives.
*   **Structured Workflow:** Follows a defined process:
    1.  Decomposition: Breaks down the request.
    2.  Consultation: Gathers suggestions from multiple models.
    3.  Analysis: Evaluates and summarizes suggestions.
    4.  Segmentation: Organizes suggestions into architectural sections.
    5.  Comparison: Analyzes relationships across steps.
    6.  Synthesis: Integrates the best ideas into a final plan.
*   **Configurable:** Allows users to specify API keys and select specific AI models for different workflow stages via `config.yaml`.
*   **Command-Line Interface:** Easy to run from the terminal.
*   **Logging:** Saves execution details to the `SuperArchitect/logs/` directory (if created).

## Workflow Details

SuperArchitect employs a multi-step workflow orchestrated by `main.py`:

1.  **Initial Planning Decomposition:** The initial user request is broken down into smaller, manageable planning steps using the `decomposition_model` defined in `config.yaml`.
    *   *Module:* `main.py` (initial call)
    *   *Status:* **Placeholder logic.** Needs implementation for effective decomposition.

2.  **Multi-Model Consultation:** Each planning step prompt is sent concurrently to the `consultation_models` listed in `config.yaml` to gather diverse perspectives.
    *   *Module:* `core/query_manager.py`

3.  **Analyzer AI Evaluation:** The responses from the consultation models for each step are evaluated by the `analyzer_model`. It aims to identify consensus points and summarize key recommendations.
    *   *Module:* `core/analysis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for robust consensus finding and summarization.

4.  **Architecture Segmentation:** The analyzer further processes the evaluated recommendations, segmenting them into standard architectural sections (e.g., components, data flow, technology stack).
    *   *Module:* `core/analysis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for accurate segmentation.

5.  **Comparative Analysis:** The segmented results are compared across the different planning steps to identify relationships, dependencies, and potential conflicts.
    *   *Module:* `core/synthesis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for meaningful comparison.

6.  **Synthesis and Integration:** Based on the comparative analysis, the best components and recommendations are selected and integrated by the `final_architect_model` into a cohesive final architectural plan or guide. A `summarizing_model` may also be used.
    *   *Module:* `core/synthesis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for effective selection and integration.

## Current State

**Important:** The core logic for several key workflow steps (Decomposition, Analysis, Segmentation, Comparison, Synthesis) is currently implemented using placeholders. These components require further development to achieve full functionality.

## Project Structure

```
SuperArchitect/
├── .gitignore
├── config.example.yaml     # Example configuration file
├── config.yaml             # User's configuration (create this file)
├── LICENSE                 # Project license
├── main.py                 # Main CLI entry point
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── USER_GUIDE.md           # Guide for non-technical users
└── core/                   # Core application logic
    ├── __init__.py
    ├── file_manager.py     # File system utilities
    ├── query_manager.py    # Handles interaction with AI models
    ├── analysis/           # Logic for analyzing model responses
    │   ├── __init__.py
    │   └── engine.py
    ├── models/             # Handlers for specific AI model APIs
    │   ├── __init__.py
    │   ├── base.py
    │   ├── claude.py
    │   ├── curl_parser.py  # Utility for parsing CURL commands (if needed)
    │   ├── gemini.py
    │   └── openai.py
    └── synthesis/          # Logic for synthesizing the final output
        ├── __init__.py
        └── engine.py
```

## Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd SuperArchitect
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create `config.yaml`:** Copy the `config.example.yaml` file to `config.yaml` in the `SuperArchitect` directory.
    ```bash
    # Windows
    copy config.example.yaml config.yaml
    # macOS/Linux
    cp config.example.yaml config.yaml
    ```
2.  **Add API Keys:** Open `config.yaml` and replace the placeholder values (`YOUR_..._API_KEY_HERE`) with your actual API keys for OpenAI, Anthropic (Claude), and Google (Gemini).
    ```yaml
    # config.yaml
    api_keys:
      openai: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      claude: sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xxxxxxxxxx
      gemini: AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
3.  **Select Models (Optional):** Review the models specified under the `models` section in `config.yaml`. Ensure these model names are correct and accessible with your API keys. You can change them based on your needs and available access.
    ```yaml
    # config.yaml (example model selection)
    models:
      decomposition_model: 'claude-3-opus-20240229'
      consultation_models:
        - 'gemini-1.5-pro-latest'
        # - 'claude-3-opus-20240229' # Add more if desired
      analyzer_model: 'claude-3-opus-20240229'
      final_architect_model: 'claude-3-opus-20240229'
      summarizing_model: 'gemini-1.5-pro-latest'
    ```

## Usage

Run the tool from the command line within the `SuperArchitect` directory:

```bash
python main.py "your architectural planning query here"
```

Replace `"your architectural planning query here"` with the actual high-level request you want the tool to process (e.g., `"Design a scalable microservices architecture for an e-commerce platform"`).

The tool will output the generated plan to the console and save execution logs (JSON format) to the `SuperArchitect/logs/` directory (this directory might need to be created manually if it doesn't exist).

Refer to `USER_GUIDE.md` for a less technical overview.

## Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflows. (Further contribution guidelines can be added here).

## License

This project is licensed under the terms of the LICENSE file.
