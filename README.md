# SuperArchitect

## Overview

SuperArchitect is a command-line tool designed to generate architectural plans from user queries. It utilizes a structured workflow involving multiple AI models, orchestrated by [`main.py`](main.py:0) and configured through [`config/config.yaml`](config/config.yaml:0), to process high-level requests and produce detailed architectural plans in Markdown format.

## Features

*   **Configuration Management:** Loads API keys and model preferences from [`config.yaml`](config/config.yaml:0).
*   **Modular LLM Interaction:** Supports various LLM providers ([`OpenAI`](core/models/openai.py:14), [`Claude`](core/models/claude.py:16), [`Gemini`](core/models/gemini.py:28)) through a common [`ModelHandler`](core/models/base.py:8) interface.
*   **Request Decomposition:** Breaks down the initial user query into logical substeps using a configured `decomposition_model`.
*   **Per-Substep Consultation:** Gathers insights for each substep by querying configured `consultation_models`.
*   **Detailed Analysis:** An [`AnalyzerEngine`](core/analysis/engine.py:36) processes consultation responses to generate structured content for each substep using an `analyzer_model`. Includes utilities for cleaning LLM-generated JSON and Markdown.
*   **Markdown Document Synthesis:** A [`SynthesisEngine`](core/synthesis/engine.py:71) assembles the final architectural plan, including headers, categorized analysis, and conclusions (optionally LLM-generated).
*   **Comprehensive Logging:** Provides detailed logging of the process in JSON format and to the console.
*   **Deep Research Capability:** (Optional) Integrates with an external tool for automated deep research on topics.

## Architecture

The SuperArchitect workflow is driven by [`main.py`](main.py:0) and proceeds through several key phases, leveraging different components of the system:

1.  **Orchestration ([`main.py`](main.py:0)):**
    *   Handles Command Line Interface (CLI) arguments.
    *   Loads configuration from [`config/config.yaml`](config/config.yaml:0), including API keys and model choices.
    *   Manages logging for the entire execution.
    *   Controls the overall process flow, initializing and coordinating model handlers and processing engines.

2.  **Model Interaction ([`core/models/`](core/models/__init__.py:0)):**
    *   A base [`ModelHandler`](core/models/base.py:8) defines a common interface for communicating with various Language Model (LLM) APIs.
    *   Specific handlers for [`OpenAI`](core/models/openai.py:14), [`Anthropic (Claude)`](core/models/claude.py:16), and [`Google (Gemini)`](core/models/gemini.py:28) implement this interface.
    *   Factory functions within [`core/models/__init__.py`](core/models/__init__.py:0) instantiate the appropriate model handlers based on the project's configuration.

3.  **Decomposition (via [`main.py`](main.py:0)):**
    *   The initial user's architectural request is processed by the `decomposition_model` (configured in [`config/config.yaml`](config/config.yaml:0)).
    *   This model breaks down the high-level query into a series of logical substeps, often formulated as questions to guide further investigation. JSON cleaning utilities from the [`AnalyzerEngine`](core/analysis/engine.py:36) may be used here.

4.  **Consultation (Per Substep, managed in [`main.py`](main.py:0)):**
    *   For each question/substep generated during decomposition, the system iterates through the list of `consultation_models` defined in [`config/config.yaml`](config/config.yaml:0).
    *   It uses the first successful response obtained from these models for further processing.

5.  **Analysis ([`core/analysis/engine.py`](core/analysis/engine.py:0)):**
    *   The [`AnalyzerEngine`](core/analysis/engine.py:36), configured with an `analyzer_model`, takes the "best answer" (the successful response from the consultation phase) for each substep.
    *   It processes this answer using an "analyzer" LLM to generate detailed content, typically structured into predefined categories (e.g., "Summary/Overview," "Key Considerations").
    *   The engine includes robust utilities for cleaning and validating LLM-generated Markdown and JSON outputs. Tested by [`core/analysis/test_json_decoding.py`](core/analysis/test_json_decoding.py:0).

6.  **Synthesis ([`core/synthesis/engine.py`](core/synthesis/engine.py:0)):**
    *   The [`SynthesisEngine`](core/synthesis/engine.py:71) assembles the final Markdown architectural document.
    *   It writes the overall project header, introduction (optionally LLM-generated), and table of contents.
    *   It then appends the categorized and formatted analysis for each substep (as provided by the [`AnalyzerEngine`](core/analysis/engine.py:36) via [`main.py`](main.py:0)).
    *   Finally, it writes the conclusion (also optionally LLM-generated).

7.  **Configuration ([`config/config.yaml`](config/config.yaml:0)):**
    *   This file is central to the project's operation, dictating model choices, API keys, logging levels, and other operational parameters.

### Key Interactions & Data Flow

*   [`main.py`](main.py:0) orchestrates the entire pipeline, initializing components and passing data between stages.
*   It takes the user query, sends it for decomposition, manages the consultation for each substep, passes results to the [`AnalyzerEngine`](core/analysis/engine.py:36) for detailed content generation, and finally provides this content to the [`SynthesisEngine`](core/synthesis/engine.py:71) for assembling the final document.
*   Model selection and API keys are sourced from [`config/config.yaml`](config/config.yaml:0) and used to initialize the correct model handlers via factories in [`core/models/__init__.py`](core/models/__init__.py:0).

### Architectural Flow Diagram

```mermaid
graph TD
    A[User Query] --> B((main.py - Orchestration));
    B -- Manages & Initializes --> C{config/config.yaml};
    B -- Uses --> D[core/models/* (Model Handlers)];
    D -- Interactions --> E[LLMs for Decomposition/Consultation/Analysis/Synthesis];
    B -- Decomposed Substeps & Consultation Data --> G[core/analysis/engine.py (AnalyzerEngine)];
    G -- Uses --> E;
    G -- Categorized Content --> I[core/synthesis/engine.py (SynthesisEngine)];
    I -- Uses --> E;
    I -- Assembles --> J[Final Architectural Plan (Markdown)];
    C -.-> D;
    C -.-> G;
    C -.-> I;
```

## Model Roles in `config.yaml`

The behavior and capabilities of SuperArchitect are significantly influenced by the AI models configured in [`config.yaml`](config/config.yaml:0) under the `model_roles` section:

*   **`decomposition_model`**: This model is responsible for the initial phase of breaking down the user's high-level request into smaller, manageable substeps or questions.
*   **`consultation_models`**: A list of language models. For each substep, SuperArchitect will attempt to get a response from these models sequentially, using the first successful answer. These models provide the core information and suggestions for each substep.
*   **`analyzer_model`**: This model is used by the [`AnalyzerEngine`](core/analysis/engine.py:36). Its role is to take the raw answer from a consultation model for a substep and transform it into a structured, detailed content output (often JSON, then formatted to Markdown).
*   **`synthesis_handler`**: This model (or handler configuration) is used by the [`SynthesisEngine`](core/synthesis/engine.py:71), primarily for generating the introductory and concluding sections of the final Markdown document, providing a narrative frame around the detailed substep guides.
*   **`summarizing_model`**: While this role may be defined in some older [`config.yaml`](config.yaml:0) versions or documentation, it is not actively used in the current primary workflow for generating the final architectural plan.

## Code Structure

The project is organized into several key modules and directories:

*   **[`main.py`](main.py:0):** The main entry point for the CLI application. It orchestrates the entire workflow, from parsing arguments to generating the final output.
*   **[`config/`](config/__init__.py:0):**
    *   [`config/config.yaml`](config/config.yaml:0): The central configuration file for API keys, model selections, model roles, logging, and other settings.
    *   [`config/config.example.yaml`](config/config.example.yaml:0): An example configuration file.
*   **[`core/`](core/__init__.py:0):** Contains the core logic of the application.
    *   **[`core/models/`](core/models/__init__.py:0):** Handles interactions with different LLM providers.
        *   [`core/models/base.py`](core/models/base.py:8): Defines the base `ModelHandler` interface.
        *   [`core/models/openai.py`](core/models/openai.py:14): Implements the handler for OpenAI models.
        *   [`core/models/claude.py`](core/models/claude.py:16): Implements the handler for Anthropic (Claude) models.
        *   [`core/models/gemini.py`](core/models/gemini.py:28): Implements the handler for Google (Gemini) models.
        *   [`core/models/__init__.py`](core/models/__init__.py:0): Contains factory functions to instantiate model handlers.
        *   [`core/models/curl_parser.py`](core/models/curl_parser.py:0): A utility for parsing cURL commands, not central to the main LLM interaction logic for plan generation.
    *   **[`core/analysis/`](core/analysis/__init__.py:0):**
        *   [`core/analysis/engine.py`](core/analysis/engine.py:0): Contains the [`AnalyzerEngine`](core/analysis/engine.py:36) responsible for processing consultation responses and generating detailed content for each substep.
        *   [`core/analysis/test_json_decoding.py`](core/analysis/test_json_decoding.py:0): Contains tests for JSON cleaning utilities.
    *   **[`core/synthesis/`](core/synthesis/__init__.py:0):**
        *   [`core/synthesis/engine.py`](core/synthesis/engine.py:71): Contains the [`SynthesisEngine`](core/synthesis/engine.py:71) responsible for assembling the final Markdown document.
    *   [`core/query_manager.py`](core/query_manager.py:10): Provides generic query execution capabilities, not directly used in the primary `run_cli` workflow for plan generation but offers underlying LLM communication utilities.
    *   `__init__.py` files in `core` and its subdirectories facilitate Python's module system and imports.
*   **[`logs/`](logs/__init__.py:0):** Directory where execution logs are stored.
*   **[`output/`](output/__init__.py:0):** Directory where generated architectural plans are saved.
*   **[`requirements.txt`](requirements.txt:0):** Lists project dependencies.
*   **[`README.md`](README.md:0):** This file.
*   **[`USER_GUIDE.md`](USER_GUIDE.md:0):** Provides additional user guidance.
*   **[`LICENSE`](LICENSE:0):** Project license information.


## Current Implementation and Future Vision

The current version of SuperArchitect implements the core pipeline described above: decomposing a request, gathering information per substep via consultation, performing detailed analysis to structure that information, and synthesizing a final document. It provides a modular way to leverage different LLMs for specialized tasks within the generation pipeline.

Previous versions of this README may have described a more ambitious workflow involving features like multi-model consensus for each question or advanced architectural segmentation. While these are not part of the current active implementation, they may represent areas for future development. The tool currently focuses on a streamlined process of generating detailed architectural guides based on configured model interactions.

## Deep Research Module

SuperArchitect includes an optional deep research module that can perform automated research using an external tool (e.g., "Auto-Deep-Research-main"). This module can be configured via the `research` section in [`config.yaml`](config/config.yaml:0). More details might be found in specific documentation if this feature is actively used (e.g., a `RESEARCH.md` file, if present).

## Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd SuperArchitect
    ```

2.  **Install Dependencies:**
    Navigate to the `SuperArchitect` directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    Create a `config.yaml` file in the `SuperArchitect` directory (you can copy and rename [`config.example.yaml`](config/config.example.yaml:0) to [`config.yaml`](config/config.yaml:0)).
    Add your API keys for the desired model providers (OpenAI, Anthropic, Google Gemini) under the `api_keys` section. Also, configure your desired models under `model_provider_models` and assign them to roles under `model_roles`.

    Example [`config.yaml`](config/config.yaml:0) structure:
    ```yaml
    api_keys:
      openai: YOUR_OPENAI_API_KEY
      anthropic: YOUR_ANTHROPIC_API_KEY
      google_gemini: YOUR_GEMINI_API_KEY

    # Defines which specific models from providers are available
    model_provider_models:
      openai_model_name: "gpt-4-turbo" # Example, use your preferred model
      anthropic_model_name: "claude-3-opus-20240229" # Example
      google_gemini_model_name: "gemini-1.5-pro-latest" # Example

    model_roles:
      decomposition_model: "google_gemini_model_name" # Assign a model from above
      consultation_models:
        - "anthropic_model_name"
        - "openai_model_name"
      analyzer_model: "google_gemini_model_name"
      synthesis_handler: # Configuration for LLM-based synthesis elements
        provider: "anthropic" # or "openai", "google_gemini"
        model_name: "claude-3-sonnet-20240229" # A model suitable for this
    # ... other configurations like logging, research etc.
    ```

## Usage

Run the tool from the command line within the `SuperArchitect` directory:

```bash
python main.py "your architectural planning query here"
```

Replace `"your architectural planning query here"` with the actual high-level request you want the tool to process. The tool will output the generated plan (e.g., to `output/generated_architectural_plan_YYYYMMDD_HHMMSS.md`) and save execution logs to the `logs/` directory.

## Configuration (`config.yaml`)

The [`config.yaml`](config/config.yaml:0) file is crucial for controlling SuperArchitect's behavior. Refer to the example above and [`config.example.yaml`](config/config.example.yaml:0) for detailed structure. Key sections include:

*   **`api_keys`**: Stores API keys for AI model providers.
*   **`model_provider_models`**: Lists specific model names available from each provider that can be referenced in `model_roles`.
*   **`model_roles`**: Assigns models (by referencing keys from `model_provider_models`) to specific roles in the workflow (see "Model Roles in `config.yaml`" section above for details).
*   **`logging`**: Configures logging levels and output paths.
*   **`research`**: (Optional) Configures the deep research module if used.