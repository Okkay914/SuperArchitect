# SuperArchitect

## Overview

SuperArchitect is a command-line tool designed to assist in architectural planning. It aims to take a high-level user request, process it through a structured workflow involving multiple AI models, and generate a detailed architectural plan in Markdown format.

## Features

*   **Configuration Management:** Loads API keys and model preferences from [`config.yaml`](config.yaml:0).
*   **Request Decomposition:** Breaks down the initial user query into logical substeps with questions using a `decomposition_model`.
*   **Per-Question Consultation:** Iterates through configured `consultation_models` for each question, using the first successful response to gather insights.
*   **Instructional Guide Generation:** An `AnalyzerEngine` (utilizing an `analyzer_model`) processes consultation responses to generate a detailed JSON instructional guide for each substep.
*   **Markdown Document Assembly:** A `SynthesisEngine` assembles the final architectural plan. This includes project headers, formatted instructional guides for each substep, and overall conclusions. An LLM (via `synthesis_handler`) can optionally generate the introduction and conclusion.
*   **Logging:** Provides comprehensive logging of the process in both JSON format and to the console.

## Workflow

The SuperArchitect workflow is orchestrated by [`main.py`](main.py:0) and proceeds through the following phases:

1.  **Decomposition:**
    *   The initial user's architectural request is processed by the `decomposition_model`.
    *   This model breaks down the high-level query into a series of logical substeps, each typically formulated as a question to guide further investigation.

2.  **Consultation (Per Question):**
    *   For each question/substep generated during decomposition, the system iterates through the list of `consultation_models` defined in [`config.yaml`](config.yaml:0).
    *   It uses the first successful response obtained from these models for further processing.

3.  **Analysis (Per Substep):**
    *   The `AnalyzerEngine`, configured with an `analyzer_model`, takes the "best answer" (the successful response from the consultation phase) for each substep.
    *   It processes this answer to generate a detailed instructional guide in JSON format, structuring the information for that specific substep.

4.  **Synthesis:**
    *   The `SynthesisEngine` takes all the generated instructional guides (one for each substep).
    *   It assembles these into a final, coherent architectural plan in Markdown format.
    *   This includes an overall project title, introduction (optionally LLM-generated), the formatted instructional guides for each substep, and a conclusion (optionally LLM-generated).

## Model Roles in `config.yaml`

The behavior and capabilities of SuperArchitect are significantly influenced by the AI models configured in [`config.yaml`](config.yaml:0) under the `model_roles` section:

*   **`decomposition_model`**: This model is responsible for the initial phase of breaking down the user's high-level request into smaller, manageable substeps or questions.
*   **`consultation_models`**: A list of language models. For each substep, SuperArchitect will attempt to get a response from these models sequentially, using the first successful answer. These models provide the core information and suggestions for each substep.
*   **`analyzer_model`**: This model is used by the `AnalyzerEngine`. Its role is to take the raw answer from a consultation model for a substep and transform it into a structured JSON instructional guide.
*   **`synthesis_handler`** (referred to as `final_architect_model` in some configurations): This model (or handler configuration) is used by the `SynthesisEngine`, primarily for generating the introductory and concluding sections of the final Markdown document, providing a narrative frame around the detailed substep guides.
*   **`summarizing_model`**: While this role may be defined in some older [`config.yaml`](config.yaml:0) versions or documentation, it is not actively used in the current primary workflow for generating the final architectural plan.

## Current Implementation and Future Vision

The current version of SuperArchitect implements the core pipeline described above: decomposing a request, gathering information per substep, structuring that information, and synthesizing a final document.

Previous versions of this README described a more ambitious workflow involving features like multi-model consensus for each question, advanced architectural segmentation beyond instructional guides, and comparative analysis across substeps. These advanced features are not part of the current active implementation but may represent areas for future development. The tool currently focuses on a streamlined process of generating instructional guides based on single successful model consultations per substep.

## Deep Research Module

SuperArchitect now includes a deep research module that can perform automated research using the "Auto-Deep-Research-main" tool. This module can be configured via the `research` section in `config.yaml`. More details can be found in the [`RESEARCH.md`](RESEARCH.md:0) file.

## Code Structure

The main components of the workflow are implemented in the following modules:

*   [`main.py`](main.py:0): The main entry point for the CLI application, orchestrates the workflow.
*   [`core/query_manager.py`](core/query_manager.py:0): Manages sending prompts to and receiving responses from the configured AI models, particularly for the consultation phase.
*   [`core/analysis/engine.py`](core/analysis/engine.py:0): Contains the `AnalyzerEngine` logic for generating instructional guides from consultation responses.
*   [`core/synthesis/engine.py`](core/synthesis/engine.py:0): Contains the `SynthesisEngine` logic for assembling the final Markdown document.
*   `core/models/`: Contains specific handlers for different model providers (OpenAI, Claude, Gemini).
*   [`config.yaml`](config.yaml:0): Central configuration file for API keys, model selections, and model roles.

## Setup

1.  **Install Dependencies:**
    Navigate to the `SuperArchitect` directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Keys:**
    Create or edit the [`config.yaml`](config.yaml:0) file in the `SuperArchitect` directory (you can copy from [`config.example.yaml`](config.example.yaml:0)). Add your API keys for the desired model providers (OpenAI, Anthropic, Google Gemini) under the `api_keys` section. Also, configure your desired models under `model_provider_models` and assign them to roles under `model_roles`.

    Example `config.yaml` structure:
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

The [`config.yaml`](config.yaml:0) file is crucial for controlling SuperArchitect's behavior. Refer to the example above and [`config.example.yaml`](config.example.yaml:0) for detailed structure. Key sections include:

*   **`api_keys`**: Stores API keys for AI model providers.
*   **`model_provider_models`**: Lists specific model names available from each provider.
*   **`model_roles`**: Assigns models (from `model_provider_models`) to specific roles in the workflow (see "Model Roles" section above for details).
*   **`logging`**: Configures logging levels and output paths.
*   **`research`**: (Optional) Configures the deep research module.