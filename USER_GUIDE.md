# SuperArchitect User Guide

This guide explains how to set up and use the SuperArchitect tool.

## What is SuperArchitect?

SuperArchitect is a command-line tool designed to assist in architectural planning. It takes a high-level user request, processes it through a structured workflow involving multiple AI models, and generates a detailed architectural plan in Markdown format.

## Prerequisites

*   **Python:** Python 3.7+ is recommended.
*   **API Keys:** You will need API keys for the AI services you intend to use (see "Getting Started: API Keys" below).

## Installation

1.  **Navigate to Project Directory:** Open your terminal and go to the `SuperArchitect` project directory.
2.  **Install Dependencies:** Run the following command to install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install libraries such as `PyYAML` (for configuration), `python-dotenv` (for managing environment variables), and specific AI provider SDKs (e.g., `openai`, `anthropic`, `google-generativeai`).

## Getting Started: API Keys

To use SuperArchitect, you need API keys to connect to the AI services it uses. These keys grant SuperArchitect permission to leverage these AI models.

The primary services that can be used include:

*   **OpenAI:** Provides models like GPT series. (Key name in `config.yaml`: `openai`)
*   **Anthropic Claude:** Provider of Claude models. (Key name in `config.yaml`: `claude`)
*   **Google Gemini:** Google's family of AI models. (Key name in `config.yaml`: `gemini`)

**How to get your keys:**

1.  **Sign Up:** Create accounts with each AI service provider you intend to use (e.g., OpenAI, Anthropic, Google).
2.  **Find the API Key Section:** Once logged in, look for sections like "API Keys," "Developer Settings," or similar in your account dashboard.
3.  **Create a New Key:** Generate a new API key as per the provider's instructions.
4.  **Copy Your Key:** Securely copy the generated key. It might only be displayed once.

**IMPORTANT:** Your API keys are sensitive credentials. **NEVER share them publicly or commit them to version control.** Keep them confidential.

## Configuration: Setting up `config.yaml`

SuperArchitect requires a configuration file named [`config.yaml`](config/config.yaml:0) to store your API keys and define which AI models to use for different tasks.

1.  **Find the Example:** In the `SuperArchitect` directory, locate the file named [`config.example.yaml`](config/config.example.yaml:0).
2.  **Make a Copy:** Create a copy of [`config.example.yaml`](config/config.example.yaml:0) in the *same* directory.
3.  **Rename the Copy:** Rename this copied file to exactly [`config.yaml`](config/config.yaml:0).

Now, open [`config.yaml`](config/config.yaml:0) with a text editor. You will need to configure the following key sections:

### 1. API Keys (`api_keys`)

This section is where you'll put the API keys you obtained.

```yaml
api_keys:
  openai: YOUR_OPENAI_API_KEY_HERE
  claude: YOUR_CLAUDE_API_KEY_HERE
  gemini: YOUR_GEMINI_API_KEY_HERE
  # Add other providers if needed, ensuring the key name matches what the application expects
```
Replace `YOUR_PROVIDER_API_KEY_HERE` with your actual API key for each service you plan to use. If you don't have a key for a particular service, you can leave its value blank or as the placeholder, but the tool won't be able to use models from that provider.

### 2. Model Roles (`model_roles`)

This crucial section assigns specific AI models to various roles within the SuperArchitect workflow. You must use the **exact model identifier strings** as provided by the LLM vendors (e.g., `'claude-3-opus-20240229'`, `'gemini-1.5-pro-latest'`, `'gpt-4-turbo'`).

```yaml
model_roles:
  decomposition_model: 'claude-3-opus-20240229' # Example
  consultation_models:
    - 'gemini-1.5-pro-latest'
    # - 'claude-3-sonnet-20240229' # Uncomment or add more
  analyzer_model: 'claude-3-opus-20240229'      # Example
  synthesis_handler:
    provider: 'anthropic'                      # "openai", "anthropic", or "google_gemini"
    model_name: 'claude-3-sonnet-20240229'     # Specific model string for this task
  summarizing_model: 'gemini-1.0-pro'           # Note: Not actively used in main workflow
```

**Explanation of Model Roles:**

*   **`decomposition_model`**:
    *   **Function:** This AI model takes your high-level architectural planning query and breaks it down into a sequence of logical substeps. Each substep is typically defined by a title and a list of specific questions that need to be answered.
    *   **Configuration:** Assign a provider-specific model string (e.g., `'claude-3-opus-20240229'`). Choose a powerful model for best results.
*   **`consultation_models`**:
    *   **Function:** This is a list of provider-specific AI model strings. For *each question* within a decomposed substep, SuperArchitect will try to get an answer from these models one by one, in the order they are listed. It uses the *first successful response* it receives as the "best answer" for that question. This allows for fallback if one model fails.
    *   **Configuration:** Provide a list of model strings (e.g., `['gemini-1.5-pro-latest', 'claude-3-sonnet-20240229']`).
*   **`analyzer_model`**:
    *   **Function:** Used by the `AnalyzerEngine`. After the "best answers" for a substep's questions are collected (from `consultation_models`), this model processes these answers (along with context) to generate a detailed, structured "instructional guide" for that substep. The output is typically a JSON object organized by predefined categories (e.g., "Summary/Overview," "Key Considerations").
    *   **Configuration:** Assign a provider-specific model string (e.g., `'claude-3-opus-20240229'`). Choose a model known for good structured output (especially JSON) and detailed, coherent responses.
*   **`synthesis_handler`**:
    *   **Function:** This configuration is used by the `SynthesisEngine`. Its primary role is to generate the introductory and concluding sections of the final Markdown document. This helps frame the detailed, structured guides (created by the `analyzer_model` for each substep) with a coherent narrative.
    *   **Configuration:** This is an object where you specify the `provider` (e.g., "openai", "anthropic", "google_gemini" – this should map to a key in your `api_keys` section) and the `model_name` (the actual model string from that provider, e.g., `'claude-3-sonnet-20240229'`).
*   **`summarizing_model`**:
    *   **Function (Legacy/Future):** This role was potentially intended for generating concise summaries.
    *   **Status:** **NOT ACTIVELY USED** in the current primary workflow for generating the final architectural plan. It can generally be ignored or removed unless you have a custom workflow that specifically utilizes it.
    *   **Configuration:** Assign a provider-specific model string.

**Save the [`config.yaml`](config/config.yaml:0) file.** Remember this file contains sensitive API keys, so keep it secure.

## SuperArchitect Workflow Overview

The tool follows a structured process to generate architectural plans:

1.  **Decomposition:** Your initial architectural query is processed by the `decomposition_model`. This model breaks down the query into a series of logical substeps, each with a title and a list of questions, output as JSON.
2.  **Consultation (Per Question):** For each question within each substep, SuperArchitect iterates through the list of `consultation_models`. It uses the first successful response obtained from these models as the best answer for that specific question.
3.  **Analysis (Per Substep):** The `AnalyzerEngine`, using the `analyzer_model`, takes all the "best answers" collected for a substep. It processes this information to generate a detailed instructional guide in a structured JSON format, categorized for clarity (e.g., "Summary/Overview," "Key Considerations").
4.  **Synthesis:** The `SynthesisEngine` collects all the categorized JSON instructional guides from each substep. It assembles them into a final Markdown architectural plan. The `synthesis_handler` (if configured) is used to generate the introduction and conclusion for this document.

## Running SuperArchitect

SuperArchitect is run from your command-line interface (Terminal, Command Prompt, PowerShell).

1.  **Open your terminal.**
2.  **Navigate to the Project Directory:** Use `cd` commands to go to the `SuperArchitect` directory where [`main.py`](main.py:0) is located.
3.  **Run the command:** Type the following, replacing the example query with your own:

    ```bash
    python main.py "your architectural planning query here"
    ```

    *Example:*
    ```bash
    python main.py "Design a scalable backend for a social media application focusing on real-time updates and media storage."
    ```

4.  **Await Output:** The tool will begin processing your query. This may take some time depending on the query's complexity and the responsiveness of the AI models.

## Expected Input and Output

*   **Input:** A textual user query (provided as a command-line argument) detailing the architectural planning need.
*   **Output:**
    *   **Architectural Plan:** A Markdown file saved in the `output/` directory (e.g., `output/generated_architectural_plan_YYYYMMDD_HHMMSS.md`).
    *   **Execution Log:** A detailed JSON log of the execution steps, saved in the `logs/` directory (e.g., `logs/execution_log_YYYYMMDD_HHMMSS.json`).
    *   **Console Log:** A file containing the full console output during the run, saved in the `logs/` directory (e.g., `logs/full_console_output_YYYYMMDD_HHMMSS.log`).

## Deep Research Module

SuperArchitect can optionally use a deep research module to gather additional context for the AI models. This feature is configured in [`config.yaml`](config/config.yaml:0). (For more details on configuring and using the research module, you might refer to a dedicated `RESEARCH.md` if available, or explore the `research` section in `config.example.yaml`).

### Enabling/Disabling in `config.yaml`

The research module is controlled via the `research` section in [`config.yaml`](config/config.yaml:0).

```yaml
research:
  enabled: true  # Set to true to enable, false to disable
  # ... other research-specific configurations
```

### Command-Line Control

You can override the `config.yaml` setting for research using command-line flags:

*   `--research-only`: Runs only the research module and then exits. Useful for pre-gathering information.
    ```bash
    python main.py "your query" --research-only
    ```
*   `--skip-research`: Skips the research module, even if enabled in `config.yaml`.
    ```bash
    python main.py "your query" --skip-research
    ```
These options provide flexibility in how you leverage the research capabilities.