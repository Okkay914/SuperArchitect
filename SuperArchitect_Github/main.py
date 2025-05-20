import asyncio
import sys # Added for stdout/stderr redirection
# SuperArchitect/main.py
import os
import yaml
import argparse
import json
from datetime import datetime
from typing import Any, List, Dict, Optional # Added List, Dict, Optional for type hinting
import traceback # Added for exception logging
import logging # Added for logging within placeholders
import aiohttp # Add import for session management

# --- Tee Class for Logging stdout/stderr ---
class Tee:
    def __init__(self, filename, mode, original_stream):
        self.file = open(filename, mode, encoding='utf-8')
        self.original_stream = original_stream

    def write(self, message):
        self.original_stream.write(message)
        self.file.write(message)
        self.flush() # Ensure immediate write to file

    def flush(self):
        self.original_stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# Assuming core components are structured like this
# Adjust imports based on actual project structure if different
from core.query_manager import QueryManager
from core.analysis.engine import AnalyzerEngine
from core.synthesis.engine import SynthesisEngine, append_substep_analysis_to_markdown
from core.models import get_model_handler_for_role, GeminiHandler, ClaudeHandler, OpenAIHandler # Import specific handlers
from core.models.base import ModelHandler # Import base handler for type hinting
# Removed import of WorkflowManager
from core.autoagent_utils.io_utils import clean_llm_json_output # Added import

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
DEFAULT_CONFIG = {
    'api_keys': {
        'openai': None,
        'claude': None,
        'gemini': None,
    },
    'models': {
        'decomposition_model': 'gemini',
        'consultation_models': ['gemini', 'claude', 'openai'],
        'analyzer_model': 'gemini',
        'final_architect_model': 'claude'
    }
}

def load_config():
    """Loads configuration from YAML file and environment variables."""
    config = DEFAULT_CONFIG.copy()

    logger.info(f"Looking for configuration file at: {CONFIG_PATH}")

    # 1. Load from YAML file
    try:
        logger.info(f"Attempting to load configuration from {CONFIG_PATH}...")
        with open(CONFIG_PATH, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for key, value in yaml_config.items():
                    if isinstance(value, dict) and isinstance(config.get(key), dict):
                        if key == 'models' and 'consultation_models' in value:
                             config[key]['consultation_models'] = value['consultation_models']
                             for model_key, model_val in value.items():
                                 if model_key != 'consultation_models':
                                     config[key][model_key] = model_val
                        else:
                            config[key].update(value)
                    else:
                        config[key] = value
                logger.info(f"Successfully loaded configuration from {CONFIG_PATH}.")
            else:
                logger.info(f"{CONFIG_PATH} is empty. Using default config and environment variables.")
    except FileNotFoundError:
        logger.warning(f"WARNING: {CONFIG_PATH} not found. Using default config and environment variables.")
    except yaml.YAMLError as e:
        logger.error(f"ERROR: Error parsing {CONFIG_PATH}: {e}. Using default config and environment variables.")
    except Exception as e:
         logger.error(f"ERROR: Unexpected error loading {CONFIG_PATH}: {e}. Using default config and environment variables.")


    # 2. Load API keys from environment variables (as fallback or if YAML value is placeholder)
    logger.info("Attempting to load API keys from environment variables (if needed)...")
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            logger.info("Successfully processed .env file (if found).")
        else:
             logger.info(".env file not found or empty, checking environment directly.")
    except ImportError:
        logger.warning("WARNING: python-dotenv not installed. API keys must be set in the environment manually if not in config.yaml.")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")

    api_keys_config = config.get('api_keys', {})
    for service in api_keys_config.keys():
        yaml_key = api_keys_config.get(service)
        if yaml_key and isinstance(yaml_key, str) and not yaml_key.startswith('YOUR_') and not yaml_key.endswith('_API_KEY_HERE'):
            logger.info(f"Using {service.upper()} API key from config.yaml.")
            api_keys_config[service] = yaml_key
        else:
            env_var_name = f'{service.upper()}_API_KEY'
            env_key = os.environ.get(env_var_name)
            if env_key:
                logger.info(f"Using {service.upper()} API key from environment variable {env_var_name}.")
                api_keys_config[service] = env_key
            else:
                if not yaml_key or (isinstance(yaml_key, str) and (yaml_key.startswith('YOUR_') or yaml_key.endswith('_API_KEY_HERE'))):
                    logger.warning(f"WARNING: {service.upper()} API key not found in config.yaml or environment variable {env_var_name}.")
                    api_keys_config[service] = None

    config['api_keys'] = api_keys_config
    logger.info("Configuration loading complete.")
    return config

# --- Execution Logging ---
execution_log = []

def log_step(description: str, details: Any = None):
    """Logs a step in the execution process."""
    timestamp = datetime.now().isoformat()
    log_entry = {"timestamp": timestamp, "description": description}
    if details is not None:
        try:
            json.dumps(details)
            log_entry["details"] = details
        except TypeError:
             log_entry["details"] = f"<{type(details).__name__} object - not JSON serializable>"
    execution_log.append(log_entry)
    logger.info(f"Log Step: {description} - Details: {log_entry.get('details', 'N/A')}")

def write_execution_log(log_data: list, run_timestamp_str: str):
    """Writes the execution log to a JSON file."""
    timestamp_str = run_timestamp_str # Use the passed timestamp for the log file name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "SuperArchitect":
        workspace_root = os.path.dirname(script_dir)
        log_dir = os.path.join(workspace_root, "SuperArchitect", "logs")
    else:
        logger.warning(f"Unexpected script location: {script_dir}. Logging to 'logs' subdirectory relative to script.")
        log_dir = os.path.join(script_dir, "logs")

    log_filename = os.path.join(log_dir, f"execution_log_{timestamp_str}.json")
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_filename, 'w') as f:
            json.dump(log_data, f, indent=4)
        logger.info(f"\nExecution log written to {log_filename}")
    except Exception as e:
        logger.error(f"\nERROR: Failed to write execution log to {log_filename}: {e}")


# --- Workflow Steps ---

async def decompose_request(user_request: str, config: dict, decomposition_handler: Optional[ModelHandler]) -> List[Dict[str, Any]]:
    """
    Step 1: Initial Planning Decomposition.
    Breaks down the user request into sequential steps using the decomposition model.

    Args:
        user_request: The high-level user request.
        config: The application configuration dictionary.
        decomposition_handler: The instantiated model handler for the decomposition model.

    Returns:
        A list of dictionaries, each representing a substep with 'step_number', 'name', and 'prompt'.
    """
    log_step("Executing Step 1: Initial Planning Decomposition")
    logger.info(f"Decomposing request: '{user_request[:50]}...' using decomposition model.")

    if not decomposition_handler or not decomposition_handler.is_ready():
        logger.error("Decomposition model handler not available or ready. Cannot decompose request.")
        # Fallback: Return a single step with the original request if decomposition fails
        logger.warning("Decomposition handler not ready, using fallback substep.")
        return [{"step_number": 1, "title": f"Initial Fallback: Process User Request '{user_request[:30]}...'", "questions": [f"What are the key aspects of '{user_request[:50]}...'?", "How can this be approached directly?"]}]

    prompt = f"""Break down the following high-level user query into a sequence of logical substeps.
For each substep:
1. Assign a sequential 'step_number'.
2. Provide a concise 'title' for the substep.
3. Generate a list of 3-5 crucial 'questions' that, if answered, would thoroughly cover the details of that substep.

Provide the output STRICTLY in JSON format, as a dictionary with a single key 'substeps', which contains a list of these substep objects.
Example:
```json
{{
  "substeps": [
    {{
      "step_number": 1,
      "title": "Define Lead Generation Objectives and Scope",
      "questions": [
        "What are the primary business goals this lead agent should achieve?",
        "How can we precisely define the Ideal Customer Profile (ICP)?",
        "What specific criteria will qualify a lead (e.g., BANT)?",
        "Which KPIs will measure success?"
      ]
    }},
    {{
      "step_number": 2,
      "title": "Identify and Profile Lead Sources",
      "questions": [
        "What are the primary online platforms for finding potential leads?",
        "What offline methods or data sources are relevant?"
      ]
    }}
  ]
}}
```

User Query:
{user_request}

JSON Response:
"""

    try:
        response = await decomposition_handler.execute(prompt)

        if response['status'] == 'success' and response['content']:
            raw_llm_output = response['content']
            # Use the new robust JSON cleaning function
            cleaned_json_string = clean_llm_json_output(raw_llm_output)
            
            if not cleaned_json_string:
                logger.error(f"Failed to extract any JSON content from LLM response after cleaning. Raw output: {raw_llm_output[:500]}")
                return [{"step_number": 1, "title": f"Process User Request (JSON cleaning failed): {user_request[:30]}...", "questions": [f"The LLM response could not be cleaned into valid JSON. What are the key aspects of '{user_request[:50]}...'?", "How should this be approached manually?"]}]

            try:
                # Attempt to parse the cleaned JSON response
                parsed_json = json.loads(cleaned_json_string)

                # Validate the structure
                if (isinstance(parsed_json, dict) and
                    'substeps' in parsed_json and
                    isinstance(parsed_json['substeps'], list) and
                    all(isinstance(item, dict) and
                        'step_number' in item and
                        'title' in item and
                        'questions' in item and
                        isinstance(item['questions'], list) and
                        all(isinstance(q, str) for q in item['questions'])
                        for item in parsed_json['substeps'])):
                    
                    substeps_list = parsed_json['substeps']
                    if not substeps_list: # Handle empty list case
                        logger.warning(f"Decomposition LLM returned an empty list of substeps. Content: {content[:500]}...")
                        return [{"step_number": 1, "title": f"Process User Request (empty substeps fallback): {user_request[:30]}...", "questions": [f"The model did not break down '{user_request[:50]}...'. What are its core components?", "What are the main goals?"]}]
                    logger.info(f"Decomposition complete. Identified {len(substeps_list)} substeps with questions.")
                    return substeps_list # Return the list of substeps
                else:
                    logger.error(f"Decomposition LLM returned invalid JSON structure or did not match expected format. Parsed: {str(parsed_json)[:300]} Cleaned Content: {cleaned_json_string[:500]}...")
                    return [{"step_number": 1, "title": f"Process User Request (invalid structure): {user_request[:30]}...", "questions": [f"What are the key aspects of '{user_request[:50]}...' to consider?", "How can this request be broken down further if automatic decomposition failed?"]}]

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response from decomposition LLM: {e}. Cleaned Content: {cleaned_json_string}")
                return [{"step_number": 1, "title": f"Process User Request (JSON error): {user_request[:30]}...", "questions": [f"What are the key aspects of '{user_request[:50]}...' to consider due to JSON parsing failure?", "How should this request be approached manually?"]}]
        else:
            logger.error(f"Decomposition LLM call failed: {response.get('error_message', 'Unknown error')}")
            return [{"step_number": 1, "title": f"Process User Request (LLM call failed): {user_request[:30]}...", "questions": [f"The decomposition model failed. What are the primary objectives of '{user_request[:50]}...'?", "What are the main deliverables?"]}]

    except Exception as e:
        logger.error(f"Error during decomposition LLM call: {e}", exc_info=True)
        return [{"step_number": 1, "title": f"Process User Request (exception): {user_request[:30]}...", "questions": [f"An exception occurred during decomposition. What are the core goals of '{user_request[:50]}...'?", "What immediate actions can be taken?"]}]


# Standard Categories for Segmentation
STANDARD_CATEGORIES = [
    "Summary/Overview",
    "Key Considerations/Factors",
    "Recommended Approach/Design",
    "Components and Structure",
    "Technical Recommendations",
    "Implementation Steps/Actions",
    "Pros and Cons/Trade-offs",
    "Further Research/Open Questions",
]

async def process_model_consultation(
    model_name: str,
    model_handler: ModelHandler,
    step_prompt: str,
    session: aiohttp.ClientSession  # Kept for signature consistency, though model_handler.execute may not use it
) -> tuple[str, Optional[str]]:
    """
    Processes a consultation step with a specific model.

    Args:
        model_name: The name of the model being consulted.
        model_handler: The instantiated model handler for the consultation.
        step_prompt: The prompt for the current substep.
        session: The aiohttp client session.

    Returns:
        A tuple containing the model name and its response content (str) or None if an error occurred.
    """
    logger.info(f"Consulting model {model_name} for substep with prompt: '{step_prompt[:70]}...'")
    log_step(f"Initiating consultation with model: {model_name}", {"prompt_excerpt": step_prompt[:100]})

    try:
        # Assuming model_handler.execute takes only the prompt,
        # and session management is handled within the handler instance.
        response = await model_handler.execute(step_prompt, session=session)

        if response and response.get('status') == 'success' and response.get('content'):
            logger.info(f"Successfully received response from {model_name}.")
            log_step(f"Consultation successful: {model_name}", {"response_length": len(response['content'])})
            return model_name, response['content']
        else:
            error_msg = response.get('error_message', 'Unknown error or empty content') if response else 'No response object received'
            logger.error(f"Model {model_name} consultation failed or returned empty/invalid content. Error: {error_msg}")
            log_step(f"Model consultation failed: {model_name}", {"error": error_msg, "response_received": bool(response)})
            return model_name, None
    except Exception as e:
        logger.error(f"Exception during model {model_name} consultation: {e}", exc_info=True)
        log_step(f"Exception in model consultation: {model_name}", {"exception_type": type(e).__name__, "exception_message": str(e)})
        return model_name, None

async def run_cli():
    """Runs the command-line interface for SuperArchitect."""

    # --- Attempt to set default encoding for stdout/stderr early ---
    # This is to prevent UnicodeEncodeError when logging to console.
    try:
        # sys is imported at the top of the file.
        # logging is imported and basicConfig is called at the module level.
        if hasattr(sys.stdout, 'reconfigure') and hasattr(sys.stderr, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            logging.info("System stdout/stderr reconfigured to UTF-8 with 'replace' error handling at the start of run_cli.")
        else:
            logging.warning("sys.stdout/stderr .reconfigure method not available. Console output may still have encoding issues. Consider setting PYTHONIOENCODING=UTF-8.")
    except Exception as e:
        logging.warning(f"Failed to reconfigure sys.stdout/stderr to UTF-8 in run_cli: {e}")
    # --- End of early reconfiguration ---

    parser = argparse.ArgumentParser(description="SuperArchitect CLI - Generate architectural plans.")
    parser.add_argument("query", help="The user's architectural planning request.")
    parser.add_argument("--research-only", action="store_true", help="Only perform the research phase")
    parser.add_argument("--skip-research", action="store_true", help="Skip the research phase")
    args = parser.parse_args()

    start_time = datetime.now()
    run_timestamp_str = start_time.strftime("%Y%m%d_%H%M%S") # Timestamp for all outputs of this run
    log_step("Application started", {"query": args.query})
    logger.info(f"\nProcessing request: '{args.query}'")

    # --- Setup comprehensive console output logging ---
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    console_log_file = None
    tee_stdout = None
    tee_stderr = None

    # Determine log directory (consistent with write_execution_log)
    script_dir_for_console_log = os.path.dirname(os.path.abspath(__file__))
    # Determine project root assuming main.py is at the root of the project directory (e.g., SuperArchitect_Bill)
    project_root = script_dir_for_console_log 

    log_dir_for_console_log = os.path.join(project_root, "logs")
    output_dir_for_plan = os.path.join(project_root, "output")
    
    os.makedirs(log_dir_for_console_log, exist_ok=True)
    os.makedirs(output_dir_for_plan, exist_ok=True)
    
    console_log_filename = os.path.join(log_dir_for_console_log, f"full_console_output_{run_timestamp_str}.log")
    output_filepath = os.path.join(output_dir_for_plan, f"generated_architectural_plan_{run_timestamp_str}.md")
    logger.info(f"Final architectural plan will be written to: {output_filepath}")


    try:
        console_log_file = open(console_log_filename, 'w', encoding='utf-8')
        tee_stdout = Tee(console_log_filename, 'a', original_stdout) # Append mode for Tee, file already opened 'w'
        tee_stderr = Tee(console_log_filename, 'a', original_stderr) # Append mode for Tee
        
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        logger.info(f"Comprehensive console output initiated. Logging to: {console_log_filename}")
        # Re-initialize logger handlers to use the new stdout/stderr if they were set up before redirection
        # This is important if basicConfig was called before redirection (which it is)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
        # The global `logger` instance will now use the new handlers targeting the Tee object.

        # Manage session for QueryManager, ResearchEngine, ResearchProcessor using async with
        async with aiohttp.ClientSession() as session:

            # 1. Load Configuration
            log_step("Loading configuration")
            config = load_config()
            if not config or 'api_keys' not in config:
                 raise ValueError("Configuration loading failed or returned invalid structure.")
            log_step("Configuration loaded", {"api_keys_loaded": list(config.get('api_keys', {}).keys()), "model_config": config.get('models', {})})

            # Instantiate Decomposition Model Handler
            decomposition_model_role = 'decomposition_model'
            decomposition_model_name = config.get('models', {}).get(decomposition_model_role)
            decomposition_handler: Optional[ModelHandler] = None
            if decomposition_model_name:
                decomposition_handler = get_model_handler_for_role(decomposition_model_role, config)
                if not decomposition_handler or not decomposition_handler.is_ready():
                    logger.warning(f"Failed to instantiate or ready decomposition model handler ({decomposition_model_name}) for role '{decomposition_model_role}'. Decomposition may fail.")
                    decomposition_handler = None
            else:
                logger.warning(f"Decomposition model role '{decomposition_model_role}' not found in config. Decomposition may fail.")


            # Instantiate Analyzer Model Handler
            analyzer_model_role = 'analyzer_model'
            analyzer_model_name = config.get('models', {}).get(analyzer_model_role)
            analyzer_handler: Optional[ModelHandler] = None
            if analyzer_model_name:
                analyzer_handler = get_model_handler_for_role(analyzer_model_role, config)
                if not analyzer_handler or not analyzer_handler.is_ready():
                    logger.warning(f"Failed to instantiate or ready analyzer model handler ({analyzer_model_name}) for role '{analyzer_model_role}'. Analysis may fail.")
                    analyzer_handler = None
            else:
                logger.warning(f"Analyzer model role '{analyzer_model_role}' not found in config. Analysis may fail.")

            # Instantiate AnalyzerEngine WITH the handler and categories
            analyzer_engine = AnalyzerEngine(config=config, analyzer_handler=analyzer_handler, standard_categories=STANDARD_CATEGORIES)

            # Instantiate Synthesis Model Handler
            synthesis_model_role = 'final_architect_model'
            synthesis_model_name = config.get('models', {}).get(synthesis_model_role)
            synthesis_handler: Optional[ModelHandler] = None
            if synthesis_model_name:
                synthesis_handler = get_model_handler_for_role(synthesis_model_role, config)
                if not synthesis_handler or not synthesis_handler.is_ready():
                    logger.warning(f"Failed to instantiate or ready synthesis model handler ({synthesis_model_name}) for role '{synthesis_model_role}'. Synthesis may fail.")
                    synthesis_handler = None
            else:
                logger.warning(f"Synthesis model role '{synthesis_model_role}' not found in config. Synthesis may fail.")

            # Instantiate SynthesisEngine WITH the handler
            synthesis_engine = SynthesisEngine(config=config, synthesis_handler=synthesis_handler)

            # Step 1: Initial Planning Decomposition
            # Pass the decomposition handler to the decompose_request function
            decomposed_steps = await decompose_request(args.query, config, decomposition_handler)

            if not decomposed_steps:
                raise ValueError("Decomposition failed to produce steps.")

            # --- Refined Iterative Workflow ---
            log_step("Starting Refined Iterative Workflow")
            logger.info("\n--- Starting Refined Iterative Workflow ---")

            # Write document header (introduction, TOC)
            substep_titles = [s.get('title', 'Unnamed Step') for s in decomposed_steps] # Changed 'name' to 'title'
            await synthesis_engine.write_document_header(output_filepath, args.query, substep_titles)
            log_step("Document header written", {"output_file": output_filepath, "num_substeps": len(substep_titles)})

            previous_substep_instructions_str: Optional[str] = None # Initialize before the loop

            # Iterate through decomposed substeps
            for step_info in decomposed_steps:
                step_number = step_info.get("step_number", "N/A")
                step_title = step_info.get("title", "Unknown Step")
                step_questions = step_info.get("questions", [])

                # Construct the step_prompt for consultation models (used for individual questions)
                # This 'step_prompt' variable itself is not directly used by the new analyzer flow,
                # but the logic to generate question_specific_prompt still relies on step_title and args.query.
                
                # Original logic for step_prompt (primarily for logging or if parts were to be reused)
                if not step_title or not step_questions:
                    logger.warning(f"Substep {step_number} ('{step_title}') is missing title or questions. Question processing might be affected.")
                    # Fallback step_prompt if needed elsewhere, though not directly for new analyzer
                    step_prompt_display = f"Address substep {step_number}: {step_title if step_title else 'Details missing for this step.'}"
                    if not step_questions and step_title:
                        step_prompt_display += "\nKey questions were not generated for this substep."
                    elif not step_title:
                         step_prompt_display = f"Address substep {step_number}: Details for this substep are missing."
                else:
                    formatted_questions_display = "\n".join([f"- {q}" for q in step_questions])
                    step_prompt_display = f"For the substep titled '{step_title}' (Step {step_number}), insights based on questions:\n{formatted_questions_display}"

                log_step(f"Processing substep {step_number}: {step_title}")
                logger.info(f"\n--- Processing Substep {step_number}: {step_title} ---")
                logger.debug(f"Context for substep {step_number} (for question generation): {step_prompt_display[:300]}...")


                # --- New Inner Loop for Processing Questions ---
                substep_best_answers: List[str] = []

                if not step_questions:
                    logger.warning(f"No questions found for substep {step_number}: '{step_title}'. Skipping question processing loop. Analyzer will receive an empty list of answers.")
                else:
                    logger.info(f"Processing {len(step_questions)} questions for substep {step_number}: '{step_title}'...")
                    for i, question_text in enumerate(step_questions):
                        current_question_number = i + 1
                        logger.info(f"  Substep {step_number}, Question {current_question_number}/{len(step_questions)}: '{question_text[:100]}...'")
                        log_step(f"Processing question {current_question_number} for substep {step_number}", {"question_text": question_text})
                        
                        best_answer_for_this_question: Optional[str] = None
                        consultation_models_list = config.get('models', {}).get('consultation_models', [])

                        if not consultation_models_list:
                            logger.warning(f"    No consultation models configured. Cannot get answer for question: '{question_text[:70]}...'")
                            best_answer_for_this_question = f"Error: No consultation models configured to answer question: {question_text}"
                        else:
                            # Prepare a more targeted prompt for each question for consultation models
                            question_specific_prompt = (
                                f"Overall User Query Context: \"{args.query}\"\n"
                                f"Current Substep Context: \"{step_title}\"\n\n"
                                f"Please provide a focused, detailed, and comprehensive answer to the following specific question:\n"
                                f"Question: \"{question_text}\"\n\n"
                                f"Your answer should be self-contained and directly address this question with actionable insights, explanations, or examples relevant to achieving the substep's goals."
                            )
                            logger.debug(f"  Prompt for question {current_question_number} to consultation models: {question_specific_prompt[:200]}...")
                            
                            for model_idx, model_name in enumerate(consultation_models_list):
                                consultation_handler: Optional[ModelHandler] = None
                                try:
                                   # Ensure API key for the base model type is available
                                   api_key_for_handler = config.get('api_keys', {}).get(model_name)
                                   if not api_key_for_handler: # Check if API key itself is missing or empty string
                                       logger.warning(f"      API key for consultation model type '{model_name}' not found or is empty in config. Skipping this model for question {current_question_number}.")
                                       continue

                                   # Get the specific model identifier string (e.g., "gpt-4o-mini")
                                   # model_name here is the key like 'openai', 'claude', 'gemini'
                                   specific_model_identifier = config.get('models', {}).get(model_name)
                                   if not specific_model_identifier or not isinstance(specific_model_identifier, str):
                                       logger.warning(f"      Specific model identifier for '{model_name}' not found or not a string in config['models']. Value: '{specific_model_identifier}'. Skipping.")
                                       continue
                                   
                                   # Instantiate handlers with correct parameters
                                   if model_name == 'gemini':
                                       # GeminiHandler expects (model: str, api_key: str | None = None)
                                       consultation_handler = GeminiHandler(model=specific_model_identifier, api_key=api_key_for_handler)
                                   elif model_name == 'claude':
                                       # ClaudeHandler expects (api_key: str | None = None, model: str = DEFAULT_MODEL)
                                       consultation_handler = ClaudeHandler(api_key=api_key_for_handler, model=specific_model_identifier)
                                   elif model_name == 'openai':
                                       # OpenAIHandler expects (api_key: str | None = None, model: str = DEFAULT_MODEL)
                                       consultation_handler = OpenAIHandler(api_key=api_key_for_handler, model=specific_model_identifier)
                                   else:
                                       logger.warning(f"      Unsupported consultation model type: '{model_name}'. Skipping for question {current_question_number}.")
                                       continue

                                   if not consultation_handler or not consultation_handler.is_ready():
                                       logger.warning(f"      Failed to initialize or ready handler for consultation model '{model_name}' (using identifier '{specific_model_identifier}'). Skipping for question {current_question_number}.")
                                       continue

                                   logger.info(f"      Consulting '{model_name}' (model {model_idx+1}/{len(consultation_models_list)}; specific: '{specific_model_identifier}') for question {current_question_number}: '{question_text[:70]}...'")
                                   
                                   # Pass the question_specific_prompt to process_model_consultation
                                   # The session object is correctly passed to process_model_consultation which passes it to handler.execute
                                   returned_model_name, response_content = await process_model_consultation(
                                       model_name, # This is the base name like 'openai'
                                       consultation_handler,
                                       question_specific_prompt, # Use the targeted prompt
                                       session # The aiohttp.ClientSession
                                   )

                                   if response_content and response_content.strip():
                                       logger.info(f"      Got valid response from '{returned_model_name}' for question {current_question_number}. Using this as the best answer.")
                                       best_answer_for_this_question = response_content.strip()
                                       log_step(f"Consultation successful for Q{current_question_number} with {returned_model_name}", {"question": question_text, "model": returned_model_name})
                                       break
                                   else:
                                       logger.info(f"      No valid response (None or empty) from '{returned_model_name}' for question {current_question_number}. Trying next model if available.")
                                       log_step(f"Consultation attempt for Q{current_question_number} with {returned_model_name} returned no/empty content", {"question": question_text, "model": returned_model_name})
                                except Exception as e_model_consult:
                                   logger.error(f"      Exception while trying to consult '{model_name}' (specific: '{specific_model_identifier if 'specific_model_identifier' in locals() else 'unknown'}') for question {current_question_number} ('{question_text[:70]}...'): {e_model_consult}", exc_info=False)
                                   logger.debug(f"Traceback for consult exception with {model_name} for Q{current_question_number}:", exc_info=True)
                                   log_step(f"Consultation exception for Q{current_question_number} with {model_name}", {"question": question_text, "model": model_name, "error": str(e_model_consult)})
                        
                            if best_answer_for_this_question is None: # If loop completed without a break
                                logger.warning(f"    No valid answer obtained from any consultation model for question {current_question_number}: '{question_text[:70]}...'")
                                best_answer_for_this_question = f"No answer obtained for question: {question_text}" # Placeholder
                            
                            substep_best_answers.append(best_answer_for_this_question)
                            log_step(f"Selected best answer for question {current_question_number} of substep {step_number}",
                                     {"question": question_text, "answer_preview": best_answer_for_this_question[:100], "answer_length": len(best_answer_for_this_question)})

                # After inner question loop (all questions for the current substep are processed)
                log_step(f"Collected {len(substep_best_answers)} best answers for substep {step_number}: '{step_title}'",
                         {"count": len(substep_best_answers),
                          "questions_original": step_questions,
                          "answers_preview": [ans[:70]+"..." if len(ans)>70 else ans for ans in substep_best_answers]})
                logger.info(f"Collected {len(substep_best_answers)} best answers for substep {step_number}: '{step_title}'.")
                if substep_best_answers:
                    for i, ans in enumerate(substep_best_answers):
                        question_for_log = step_questions[i] if i < len(step_questions) else "N/A"
                        logger.debug(f"  Best Answer for Q{i+1} ('{question_for_log[:60]}...'): '{ans[:150]}...'")
                else:
                     logger.info(f"  No best answers were collected for substep {step_number}.")

                # Step 3: Analyze and Synthesize (per substep using new AnalyzerEngine method)
                analysis_result = await analyzer_engine.analyze_and_synthesize_step(
                    substep_title=step_title,
                    key_question_answers=substep_best_answers,
                    overall_query_context=args.query,
                    previous_substep_instructions=previous_substep_instructions_str
                )

                if analysis_result and analysis_result.get('status') == 'success':
                    instructional_guide = analysis_result.get('instructional_guide')
                    logger.info(f"Instructional guide generated successfully for substep {step_number}: {step_title}")
                    log_step(f"Instructional guide generation successful for substep {step_number}", {"guide_keys": list(instructional_guide.keys()) if instructional_guide else "None"})
                    
                    if instructional_guide and isinstance(instructional_guide, dict):
                        try:
                            previous_substep_instructions_str = yaml.dump(instructional_guide, sort_keys=False, allow_unicode=True, indent=2)
                            logger.debug(f"Stored YAML for substep {step_number} as previous instructions context (length: {len(previous_substep_instructions_str)}).")
                        except yaml.YAMLError as ye:
                            logger.error(f"Could not dump instructional guide to YAML for context passing: {ye}")
                            previous_substep_instructions_str = f"Error: Could not serialize previous step's guide due to YAML error: {ye}"
                    else:
                        previous_substep_instructions_str = "# Previous substep generated no valid instructional guide content or was not a dictionary."
                        logger.warning(f"Instructional guide for substep {step_title} was not a dict or was None. previous_substep_instructions_str set to placeholder.")


                    # Step 4: Append to Markdown (using the new instructional_guide)
                    if synthesis_engine and synthesis_engine.is_ready():
                        await synthesis_engine.append_substep_analysis_to_markdown(
                            output_filepath,
                            step_number,
                            step_title,
                            instructional_guide
                        )
                        log_step(f"Appended substep {step_number} instructional guide to Markdown", {"output_file": output_filepath})
                    else:
                        logger.error(f"SynthesisEngine not ready. Cannot append instructional guide for substep {step_number} to Markdown.")
                        # Potentially store previous_substep_instructions_str to a file or log it extensively if markdown fails
                else:
                    error_msg = analysis_result.get('error_message', 'Instructional guide generation failed or produced no content') if analysis_result else "Analysis_result was None"
                    logger.error(f"Instructional guide generation failed for substep {step_number}: {step_title}. Error: {error_msg}")
                    log_step(f"Instructional guide generation failed for substep {step_number}", {"error": error_msg})
                    previous_substep_instructions_str = f"# Analysis for substep {step_title} failed. Error: {error_msg}"
                    
                    if synthesis_engine and synthesis_engine.synthesis_handler and synthesis_engine.synthesis_handler.is_ready():
                        await synthesis_engine.append_error_to_markdown(
                            output_filepath,
                            step_title,
                            f"Instructional guide generation for this substep failed. Error: {error_msg}"
                        )
                        log_step(f"Appended error for substep {step_number} to Markdown", {"output_file": output_filepath})
                    else:
                        logger.error(f"SynthesisEngine not ready. Cannot append error for substep {step_number} to Markdown.")

            # Write document conclusion
            await synthesis_engine.write_document_conclusion(output_filepath, args.query)
            log_step("Document conclusion written", {"output_file": output_filepath})

            logger.info(f"\n--- Incremental Architectural Plan Generation Complete ---")
            logger.info(f"Final architectural plan has been incrementally written to: {output_filepath}")
            log_step("Incremental plan generation complete", {"filename": output_filepath})

            # --- Workflow Complete ---
            logger.info("\n--- Workflow Complete ---")
            end_time = datetime.now()
            duration = end_time - start_time
            log_step("Application finished", {"duration_seconds": duration.total_seconds()})
            logger.info(f"Total execution time: {duration}")


    except Exception as e:
        detailed_error = traceback.format_exc()
        log_step("Application failed", {"error": str(e), "traceback": detailed_error})
        # sys.stderr is already our Tee object here, or original_stderr if redirection failed early
        current_stderr = sys.stderr
        print(f"\nCRITICAL ERROR: Application failed: {e}", file=current_stderr)
        print(f"\n--- DETAILED TRACEBACK ---\n{detailed_error}\n--- END TRACEBACK ---", file=current_stderr)
        # Also log to logger, which goes to Tee (if active) or original stdout (if restored)
        logger.error(f"\nCRITICAL ERROR: Application failed: {e}", exc_info=False) # exc_info=False to avoid duplicate traceback via logger if already printed

    finally:
        # Restore original stdout/stderr and close console log file
        if sys.stdout is tee_stdout and tee_stdout is not None: # Check if still redirected
            sys.stdout = original_stdout
            if hasattr(tee_stdout, 'file') and not tee_stdout.file.closed:
                tee_stdout.close()
            # logger.info after this will go to original_stdout
            if original_stdout: # Ensure original_stdout is not None
                 original_stdout.write(f"Restored stdout. Console log (stdout part) processing finished for: {console_log_filename}\n")
        elif sys.stdout is not original_stdout and original_stdout is not None: # If redirected to something else unexpectedly
             sys.stdout = original_stdout # Force restore
             original_stdout.write(f"Forcefully restored stdout.\n")


        if sys.stderr is tee_stderr and tee_stderr is not None: # Check if still redirected
            sys.stderr = original_stderr
            if hasattr(tee_stderr, 'file') and not tee_stderr.file.closed:
                tee_stderr.close()
            if original_stderr: # Ensure original_stderr is not None
                original_stderr.write(f"Restored stderr. Console log (stderr part) processing finished for: {console_log_filename}\n")
        elif sys.stderr is not original_stderr and original_stderr is not None: # If redirected to something else
            sys.stderr = original_stderr # Force restore
            original_stderr.write(f"Forcefully restored stderr.\n")
        
        # Reconfigure logging to point to original stdout for final messages
        # Ensure logging is reconfigured to a valid stream, even if original_stdout was None initially (e.g. in a non-console environment)
        final_log_stream = original_stdout if original_stdout else sys.__stdout__ # Fallback to sys.__stdout__
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            if hasattr(handler, 'close'): # Check if handler has close method
                try:
                    handler.close()
                except Exception as e_close:
                    if final_log_stream: # Check if final_log_stream is usable
                        print(f"Error closing log handler: {e_close}", file=final_log_stream)

        # Only reconfigure basicConfig if final_log_stream is valid (not None)
        if final_log_stream:
            try:
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=final_log_stream)
                logger.info("Logging reconfigured to final stream.") # This will now go to original stdout/stderr
            except Exception as e_reconfig: # Catch potential errors during reconfig (e.g. if stream is invalid)
                if final_log_stream: # Check again, though it should be the same as above
                     print(f"CRITICAL: Failed to reconfigure logging to final stream: {e_reconfig}", file=final_log_stream)
        else:
            print("CRITICAL: No valid final stream for logging. Some final log messages might be lost.", file=sys.__stderr__)


        # Write the overall execution log at the very end
        if execution_log: # Ensure there's something to write
            write_execution_log(execution_log, run_timestamp_str)
            if original_stdout: # If original_stdout is available, print confirmation
                original_stdout.write(f"Execution log processing finished. Log saved for run {run_timestamp_str}.\n")
        else:
            if original_stdout:
                original_stdout.write(f"No execution log entries to write for run {run_timestamp_str}.\n")


if __name__ == '__main__':
    # Setup to catch and log unhandled exceptions from the asyncio loop as well
    _original_stdout_main = sys.stdout # Capture current stdout before any potential asyncio changes
    _original_stderr_main = sys.stderr # Capture current stderr

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_cli())
    except Exception as e:
        # This block might catch exceptions from run_cli if they aren't handled internally,
        # or from the asyncio loop setup/teardown itself.
        detailed_error_main = traceback.format_exc()
        # Attempt to use the global logger first, which might be Tee'd
        try:
            logger.critical(f"Unhandled exception in main execution: {e}\n{detailed_error_main}", exc_info=False)
        except Exception as log_e:
            # If logger fails, fall back to original stderr
            print(f"CRITICAL (logger failed): Unhandled exception in main: {e}\n{detailed_error_main}\nLogging error: {log_e}", file=_original_stderr_main)

        # Ensure the error is also printed to the original stderr if possible
        if _original_stderr_main:
            print(f"\nCRITICAL ERROR (main level): Application terminated due to an unhandled exception: {e}", file=_original_stderr_main)
            print(f"\n--- MAIN LEVEL DETAILED TRACEBACK ---\n{detailed_error_main}\n--- END MAIN LEVEL TRACEBACK ---", file=_original_stderr_main)
        
        # Attempt to write execution log if it hasn't been (e.g., error before normal finally block)
        # This is a best-effort, as run_timestamp_str might not be defined if error was very early
        try:
            # Check if run_timestamp_str is defined; if not, generate a fallback.
            # This is tricky because run_timestamp_str is defined inside run_cli.
            # For simplicity, we'll just check if execution_log has entries.
            if execution_log: # If there are log entries, assume run_timestamp_str might be available implicitly via prior calls.
                 # We need run_timestamp_str, which is local to run_cli. This final log write here might not have it.
                 # A more robust solution would be to pass run_timestamp_str out or make it global,
                 # or have write_execution_log generate one if not provided.
                 # For now, this final attempt to write log might fail if run_timestamp_str is needed by write_execution_log
                 # and wasn't made accessible. Let's assume write_execution_log can handle it or it's written by run_cli's finally.
                 # The finally block in run_cli should handle writing the log.
                 # This is a fallback for very early/critical exits.
                 print("Attempting to write execution log due to main level exception...", file=_original_stderr_main)
                 # write_execution_log(execution_log, "CRITICAL_EXIT_" + datetime.now().strftime("%Y%m%d_%H%M%S")) # Example with fallback ts
            else:
                print("No execution log entries to write from main level exception handler.", file=_original_stderr_main)

        except Exception as final_log_e:
            print(f"Error during final attempt to write execution log: {final_log_e}", file=_original_stderr_main)
    finally:
        loop.close()
        if _original_stdout_main:
             _original_stdout_main.write("Asyncio event loop closed.\nApplication shutdown complete.\n")
        else: # Fallback if original_stdout_main was None
            sys.__stdout__.write("Asyncio event loop closed.\nApplication shutdown complete.\n")