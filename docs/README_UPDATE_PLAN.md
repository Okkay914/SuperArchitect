# Plan to Update USER_GUIDE.md

**Overall Goal:** To make the [`USER_GUIDE.md`](USER_GUIDE.md:0) more accurate, consistent with the current project structure (especially [`config.example.yaml`](config/config.example.yaml:0)), and fully comprehensive based on the provided script analysis.

**I. Review and Enhance Existing Sections:**

1.  **Introduction / What is SuperArchitect?**
    *   **Action:** Ensure this aligns with the core functionality described in the script analysis. (Current version seems good).

2.  **Getting Started: API Keys**
    *   **Action:** This section is generally good. Confirm that the provider names mentioned (OpenAI, Anthropic, Google) match the keys in [`config.example.yaml`](config/config.example.yaml:0) (`openai`, `claude`, `gemini`).

3.  **Installation / Prerequisites**
    *   **Action:**
        *   Explicitly state that Python 3.x is required (e.g., "Python 3.7+ recommended").
        *   The command `pip install -r requirements.txt` is correct.
        *   Briefly mention that this installs necessary libraries like `PyYAML` for configuration, `python-dotenv` for environment variable management, and specific AI provider libraries (`openai`, `anthropic`, `google-generativeai`).

**II. Major Update: Configuration (`config.yaml`) Section**

This section requires the most significant changes to align with [`config.example.yaml`](config/config.example.yaml:0) and the script analysis.

1.  **Remove "Model Provider Models (`model_provider_models`)" Sub-section:**
    *   **Reasoning:** The [`config.example.yaml`](config/config.example.yaml:0) shows that `model_roles` now directly use provider-specific model strings (e.g., `"claude-3-opus-20240229"`), not friendly names defined in a separate `model_provider_models` mapping.
    *   **Action:** Delete the entire existing sub-section "2. Model Provider Models (`model_provider_models`)" from [`USER_GUIDE.md`](USER_GUIDE.md:59).

2.  **Revise and Rename "Model Roles (`model_roles`)" Sub-section:**
    *   **Action:** This will become the primary model configuration sub-section after API keys.
    *   **Key Explanation to Add/Modify:**
        *   Clearly state that users must provide the **exact model identifier strings** as provided by the LLM vendors (e.g., `'claude-3-opus-20240229'`, `'gemini-1.5-pro-latest'`, `'gpt-4-turbo'`).
        *   Update the introductory text of this section to reflect this direct usage of model strings.
    *   **Specific Model Role Descriptions (align with `config.example.yaml` and script analysis):**
        *   **`decomposition_model`**:
            *   Confirm purpose: Breaks down the high-level user query into logical substeps (title + questions).
            *   Configuration: Assign a provider-specific model string.
        *   **`consultation_models`**:
            *   Confirm purpose: A list of provider-specific model strings tried sequentially for *each question* within a substep. The first successful response is used.
            *   Configuration: A list of model strings.
        *   **`analyzer_model`**:
            *   Confirm purpose: Used by [`AnalyzerEngine`](core/analysis/engine.py:0) to process "best answers" for a substep's questions into a structured JSON instructional guide (organized by categories).
            *   Configuration: Assign a provider-specific model string.
        *   **`synthesis_handler`**:
            *   **Confirm per user feedback:** This is a structured object with `provider` (e.g., "openai", "anthropic", "google_gemini") and `model_name` (the specific model string for that provider) sub-keys.
            *   Confirm purpose: Used by [`SynthesisEngine`](core/synthesis/engine.py:0) to generate introductory and concluding sections of the final Markdown document.
        *   **`summarizing_model`**:
            *   **Action:** Add a brief mention of this role as seen in [`config.example.yaml`](config/config.example.yaml:94).
            *   Clarify that it's **not actively used** in the current primary workflow and can generally be ignored or removed unless the user has a custom workflow.

**III. SuperArchitect Workflow Section**

*   **Action:** Review this section against the script analysis and the updated configuration details.
    1.  **Decomposition:** User query -> `decomposition_model` -> Substeps JSON (title, questions).
    2.  **Consultation (Per Question):** For each question -> `consultation_models` (list, sequential trial) -> Best answer for that question.
    3.  **Analysis (Per Substep):** Collected answers for substep -> [`AnalyzerEngine`](core/analysis/engine.py:0) with `analyzer_model` -> Categorized JSON guide for the substep.
    4.  **Synthesis:** All JSON guides -> [`SynthesisEngine`](core/synthesis/engine.py:0) (using `synthesis_handler` for intro/conclusion) -> Final Markdown document.
    *   Ensure terminology (e.g., "substeps," "questions") is consistent.

**IV. Running SuperArchitect Section**

*   **Action:**
    *   **Command:** `python main.py "your architectural planning query here"` is correct.
    *   **Command-Line Arguments:** Ensure clear explanations for:
        *   `query` (positional, mandatory): The main architectural request.
        *   `--research-only`: Runs only the research module and exits.
        *   `--skip-research`: Skips the research module.
    *   **Output:**
        *   Confirm mention of the output file path: `output/generated_architectural_plan_YYYYMMDD_HHMMSS.md`.
        *   Confirm mention of log files:
            *   `logs/execution_log_YYYYMMDD_HHMMSS.json` (structured log).
            *   `logs/full_console_output_YYYYMMDD_HHMMSS.log` (raw console output).

**V. Input and Output (Consolidate/Clarify if needed)**

*   **Action:** Consider a small, dedicated "Expected Input and Output" summary section if it enhances clarity.
    *   **Input:** A textual user query detailing the architectural planning need.
    *   **Output:** A Markdown file containing the generated architectural plan, plus JSON and text log files.

**VI. Mermaid Diagram for Workflow Visualization**

```mermaid
graph TD
    A[User Query] --> B{main.py};
    B -- Configures with --> C[config.yaml];
    C -- Defines --> D[API Keys];
    C -- Defines --> E[Model Roles: \n- decomposition_model\n- consultation_models\n- analyzer_model\n- synthesis_handler];
    B -- 1. Decomposition<br>(decomposition_model) --> F[Substeps + Questions JSON];
    F --> G{Loop per Question in Substep};
    G -- 2. Consultation<br>(consultation_models) --> H[Best Answer for Question];
    H -- Collect All Answers --> I[Answers for Substep];
    I --> J{core/analysis/engine.py};
    J -- 3. Analysis<br>(analyzer_model) --> K[Categorized JSON<br>for Substep];
    K -- Collect All JSON Guides --> L{core/synthesis/engine.py};
    L -- 4. Synthesis<br>(synthesis_handler for Intro/Conclusion) --> M[Final Markdown Document];
    M --> N[output/generated_architectural_plan_....md];
    B -- Logs to --> O[logs/...];