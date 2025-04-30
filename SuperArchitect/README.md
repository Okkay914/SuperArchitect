# SuperArchitect

## Overview

SuperArchitect is a command-line tool designed to assist with architectural planning by leveraging multiple AI models. It takes a high-level architectural request, processes it through a structured workflow involving decomposition, multi-model consultation, analysis, and synthesis, ultimately generating a proposed architectural plan.

## Workflow

SuperArchitect employs a 6-step workflow:

1.  **Initial Planning Decomposition:** The initial user request is broken down into smaller, manageable planning steps.
    *   *Module:* `main.py`
    *   *Status:* **Placeholder logic.** Needs implementation for effective decomposition.

2.  **Multi-Model Consultation:** Each planning step prompt is sent concurrently to multiple configured large language models (e.g., Claude, Gemini, OpenAI) to gather diverse perspectives and recommendations.
    *   *Module:* `core/query_manager.py`

3.  **Analyzer AI Evaluation:** The responses from the different models for each step are evaluated by an analyzer AI. It aims to identify consensus points and summarize the key recommendations.
    *   *Module:* `core/analysis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for robust consensus finding and summarization.

4.  **Architecture Segmentation:** The analyzer further processes the evaluated recommendations, segmenting them into standard architectural sections (e.g., components, data flow, technology stack).
    *   *Module:* `core/analysis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for accurate segmentation.

5.  **Comparative Analysis:** The segmented results are compared across the different planning steps to identify relationships, dependencies, and potential conflicts.
    *   *Module:* `core/synthesis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for meaningful comparison.

6.  **Synthesis and Integration:** Based on the comparative analysis, the best components and recommendations are selected and integrated into a cohesive final architectural plan, potentially including guidance and rationale.
    *   *Module:* `core/synthesis/engine.py`
    *   *Status:* **Placeholder logic.** Needs implementation for effective selection and integration.

## Current State

**Important:** The core logic for several key workflow steps is currently implemented using placeholders:
*   Step 1: Decomposition
*   Step 3: Analysis (Consensus, Summary)
*   Step 4: Segmentation
*   Step 5: Comparison
*   Step 6: Synthesis (Selection, Integration)

These components require further development to achieve full functionality.

## Code Structure

The main components of the workflow are implemented in the following modules:

*   `main.py`: The main entry point for the CLI application, orchestrates the workflow.
*   `core/query_manager.py`: Manages sending prompts to and receiving responses from the configured AI models.
*   `core/analysis/engine.py`: Contains the logic for evaluating and segmenting model responses (currently placeholder).
*   `core/synthesis/engine.py`: Contains the logic for comparing segments and synthesizing the final plan (currently placeholder).

## Setup

1.  **Install Dependencies:**
    Navigate to the `SuperArchitect` directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Keys:**
    Edit the `config.yaml` file located in the `SuperArchitect` directory. Add your API keys for the desired model providers (OpenAI, Anthropic, Google Gemini) under the `api_keys` section.

    ```yaml
    api_keys:
      openai: YOUR_OPENAI_API_KEY
      anthropic: YOUR_ANTHROPIC_API_KEY
      google_gemini: YOUR_GEMINI_API_KEY
    ```

## Usage

Run the tool from the command line within the `SuperArchitect` directory:

```bash
python main.py "your architectural planning query here"
```

Replace `"your architectural planning query here"` with the actual high-level request you want the tool to process. The tool will output the generated plan and save execution logs to the `SuperArchitect/logs/` directory.

## Configuration (`config.yaml`)

The `config.yaml` file controls the behavior of SuperArchitect.

### `api_keys`

Stores the API keys required to authenticate with the different AI model providers.

### `models`

Defines which specific models (e.g., `gpt-4`, `claude-3-opus-20240229`, `gemini-1.5-pro-latest`) are used by the query manager. Ensure the model names listed here are accessible via your configured API keys.

```yaml
models:
  openai: gpt-4 # Example
  claude: claude-3-opus-20240229 # Example
  gemini: gemini-1.5-pro-latest # Example
### Model Configuration Roles

The `config.yaml` file also defines the specific roles each model plays in the SuperArchitect workflow. These roles are assigned under the `model_roles` key:

*   **`decomposition_model`**: Responsible for breaking down the initial user query into logical sub-steps or planning phases.
*   **`consultation_models`**: A list of one or more models that provide initial architectural suggestions or solutions for each sub-step defined by the decomposition model.
*   **`analyzer_model`**: Evaluates the responses from the consultation models for a given sub-step, summarizes findings, identifies consensus/disagreement, and segments the recommendations into standard architectural sections.
*   **`final_architect_model`**: Synthesizes the analyzed results from all sub-steps into a single, coherent, final architectural plan or guide.
*   **`summarizing_model`**: Used for generating concise summaries, potentially of intermediate steps or the final plan.