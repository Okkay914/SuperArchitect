import asyncio
# SuperArchitect/main.py
import os
import yaml
import argparse
import sys
import json
from datetime import datetime
from typing import Any, List, Dict, Optional # Added List, Dict, Optional for type hinting
import traceback # Added for exception logging
import logging # Added for logging within placeholders
import aiohttp # Add import for session management

# Assuming core components are structured like this
# Adjust imports based on actual project structure if different
from core.query_manager import QueryManager
from core.analysis.engine import AnalyzerEngine
from core.synthesis.engine import SynthesisEngine
from core.models import get_model_handler_for_role, GeminiHandler, ClaudeHandler, OpenAIHandler # Import specific handlers
from core.models.base import ModelHandler # Import base handler for type hinting

# Configure logging for placeholders
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
# Define CONFIG_PATH relative to the script's location, not the current working directory
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
DEFAULT_CONFIG = {
    'api_keys': {
        'openai': None,
        'claude': None,
        'gemini': None,
    },
    'models': {
        # Define roles for the new workflow
        'decomposition_model': 'gemini', # Model for Step 1
        'consultation_models': ['gemini', 'claude', 'openai'], # Models for Step 2
        'analyzer_model': 'gemini', # Model for Step 3 & 4
        'final_architect_model': 'claude' # Model for Step 5 & 6
    }
}

def load_config():
    """Loads configuration from YAML file and environment variables."""
    config = DEFAULT_CONFIG.copy()

    # Log the path being used to find the config file
    logger.info(f"Looking for configuration file at: {CONFIG_PATH}")

    # 1. Load from YAML file
    try:
        logger.info(f"Attempting to load configuration from {CONFIG_PATH}...")
        with open(CONFIG_PATH, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Deep merge YAML config into defaults
                for key, value in yaml_config.items():
                    if isinstance(value, dict) and isinstance(config.get(key), dict):
                        # Special handling for models list to overwrite, not merge
                        if key == 'models' and 'consultation_models' in value:
                             config[key]['consultation_models'] = value['consultation_models']
                             # Update other model keys individually
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
        # Use dotenv only if available
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
        # Use YAML key if it's present and not the placeholder
        if yaml_key and isinstance(yaml_key, str) and not yaml_key.startswith('YOUR_') and not yaml_key.endswith('_API_KEY_HERE'):
             logger.info(f"Using {service.upper()} API key from config.yaml.")
             api_keys_config[service] = yaml_key
        else:
            # Otherwise, try environment variable
            env_var_name = f'{service.upper()}_API_KEY'
            env_key = os.environ.get(env_var_name)
            if env_key:
                logger.info(f"Using {service.upper()} API key from environment variable {env_var_name}.")
                api_keys_config[service] = env_key
            else:
                # Only warn if the key was *expected* (i.e., a placeholder was in YAML or it wasn't set)
                if not yaml_key or (isinstance(yaml_key, str) and (yaml_key.startswith('YOUR_') or yaml_key.endswith('_API_KEY_HERE'))):
                    logger.warning(f"WARNING: {service.upper()} API key not found in config.yaml or environment variable {env_var_name}.")
                api_keys_config[service] = None # Ensure it's None if not found

    config['api_keys'] = api_keys_config # Update the main config dict

    logger.info("Configuration loading complete.")
    return config

# --- Execution Logging ---
execution_log = []

def log_step(description: str, details: Any = None):
    """Logs a step in the execution process."""
    timestamp = datetime.now().isoformat()
    log_entry = {"timestamp": timestamp, "description": description}
    if details is not None:
        # Basic serialization for common types, expand as needed
        try:
            # Attempt to serialize directly
            json.dumps(details)
            log_entry["details"] = details
        except TypeError:
             log_entry["details"] = f"<{type(details).__name__} object - not JSON serializable>" # Placeholder for non-serializable
    execution_log.append(log_entry)
    logger.info(f"Log Step: {description} - Details: {log_entry.get('details', 'N/A')}") # Also log to console logger

def write_execution_log(log_data: list):
    """Writes the execution log to a JSON file."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construct path relative to the workspace root (one level above the script's directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Handle potential edge case where script is not in expected location
    if os.path.basename(script_dir) == "SuperArchitect":
        workspace_root = os.path.dirname(script_dir)
        log_dir = os.path.join(workspace_root, "SuperArchitect", "logs") # Target: C:\...\Desktop\SuperArchitect\logs
    else:
        # Fallback if structure is unexpected, log relative to script dir
        logger.warning(f"Unexpected script location: {script_dir}. Logging to 'logs' subdirectory relative to script.")
        log_dir = os.path.join(script_dir, "logs")

    log_filename = os.path.join(log_dir, f"execution_log_{timestamp_str}.json")
    try:
        os.makedirs(log_dir, exist_ok=True) # Use the calculated log_dir
        with open(log_filename, 'w') as f:
            json.dump(log_data, f, indent=4)
        logger.info(f"\nExecution log written to {log_filename}")
    except Exception as e:
        logger.error(f"\nERROR: Failed to write execution log to {log_filename}: {e}")


# --- Workflow Step Placeholders ---

async def decompose_request(user_request: str, config: dict) -> List[Dict[str, Any]]:
    """
    Placeholder for Step 1: Initial Planning Decomposition.
    Breaks down the user request into sequential steps.
    """
    log_step("Executing Step 1: Initial Planning Decomposition (Placeholder)")
    logger.info(f"[Placeholder] Decomposing request: '{user_request[:50]}...'")
    # In a real implementation, this would likely call an LLM
    # For now, return dummy steps based on keywords or simple logic
    steps = [
        {"step_number": 1, "name": "Define Core Requirements", "prompt": f"Define the core requirements based on: {user_request}"},
        {"step_number": 2, "name": "Design Data Flow", "prompt": f"Design the data flow for: {user_request}"},
        {"step_number": 3, "name": "Select Technology Stack", "prompt": f"Select the technology stack for: {user_request}"},
        {"step_number": 4, "name": "Outline Deployment Strategy", "prompt": f"Outline the deployment strategy for: {user_request}"},
    ]
    log_step("Decomposition complete (Placeholder)", {"steps_generated": len(steps)})
    return steps

# --- CLI Application Logic ---

async def run_cli(): # Keep async
    """Runs the command-line interface for SuperArchitect."""
    parser = argparse.ArgumentParser(description="SuperArchitect CLI - Generate architectural plans.")
    parser.add_argument("query", help="The user's architectural planning request.")
    args = parser.parse_args()

    start_time = datetime.now()
    log_step("Application started", {"query": args.query})
    logger.info(f"\nProcessing request: '{args.query}'")

    try:
        # Manage session for QueryManager using async with
        async with aiohttp.ClientSession() as session:

            # 1. Load Configuration
            log_step("Loading configuration")
            config = load_config()
            if not config or 'api_keys' not in config:
                 raise ValueError("Configuration loading failed or returned invalid structure.")
            log_step("Configuration loaded", {"api_keys_loaded": list(config.get('api_keys', {}).keys()), "model_config": config.get('models', {})})

            # --- NEW WORKFLOW ORCHESTRATION ---
            all_step_analysis_results = []
            query_manager = QueryManager() # Instantiate QueryManager

            # Instantiate Analyzer Model Handler
            analyzer_model_role = 'analyzer_model'
            analyzer_model_name = config.get('models', {}).get(analyzer_model_role)
            analyzer_handler: Optional[ModelHandler] = None
            if analyzer_model_name:
                 analyzer_handler = get_model_handler_for_role(analyzer_model_role, config) # Use factory
                 if not analyzer_handler or not analyzer_handler.is_ready():
                      logger.warning(f"Failed to instantiate or ready analyzer model handler ({analyzer_model_name}) for role '{analyzer_model_role}'. Analysis may fail.")
                      analyzer_handler = None # Ensure it's None if failed
            else:
                 logger.warning(f"Analyzer model role '{analyzer_model_role}' not found in config. Analysis may fail.")

            # Instantiate AnalyzerEngine WITH the handler
            analyzer_engine = AnalyzerEngine(config=config, analyzer_handler=analyzer_handler)

            # Instantiate Synthesis Model Handler
            synthesis_model_role = 'final_architect_model'
            synthesis_model_name = config.get('models', {}).get(synthesis_model_role)
            synthesis_handler: Optional[ModelHandler] = None
            if synthesis_model_name:
                synthesis_handler = get_model_handler_for_role(synthesis_model_role, config) # Use factory
                if not synthesis_handler or not synthesis_handler.is_ready():
                     logger.warning(f"Failed to instantiate or ready synthesis model handler ({synthesis_model_name}) for role '{synthesis_model_role}'. Synthesis may fail.")
                     synthesis_handler = None # Ensure it's None if failed
            else:
                logger.warning(f"Synthesis model role '{synthesis_model_role}' not found in config. Synthesis may fail.")

            # Instantiate SynthesisEngine WITH the handler
            synthesis_engine = SynthesisEngine(config=config, synthesis_handler=synthesis_handler)

            # Step 1: Initial Planning Decomposition
            decomposed_steps = await decompose_request(args.query, config)

            if not decomposed_steps:
                raise ValueError("Decomposition failed to produce steps.")

            # Loop through decomposed steps for Steps 2 & 3
            for step_data in decomposed_steps:
                step_number = step_data.get("step_number", "N/A")
                step_name = step_data.get("name", "Unnamed Step")
                step_prompt = step_data.get("prompt", "")
                log_step(f"Starting processing for Step {step_number}: {step_name}", {"prompt": step_prompt[:50] + "..."})
                logger.info(f"\n--- Processing Step {step_number}: {step_name} ---")

                if not step_prompt:
                    log_step(f"Skipping Step {step_number}", {"reason": "No prompt provided"})
                    logger.warning(f"Skipping step {step_number} due to missing prompt.")
                    all_step_analysis_results.append({
                        "step_number": step_number,
                        "step_name": step_name,
                        "status": "error",
                        "error_message": "Step prompt was missing during decomposition."
                    })
                    continue

                # Step 2: Multi-Model Consultation
                log_step(f"Executing Step 2: Multi-Model Consultation for Step {step_number}", {"step_prompt": step_prompt[:50] + "..."})
                consultation_model_names = config.get('models', {}).get('consultation_models', [])
                consultation_handlers: List[ModelHandler] = []
                logger.info(f"Instantiating handlers for consultation models: {consultation_model_names}")
                for specific_model_name in consultation_model_names:
                    handler: Optional[ModelHandler] = None
                    api_key = None
                    try:
                        # Determine base type and get API key
                        if specific_model_name.startswith('gemini'):
                            api_key = config.get('api_keys', {}).get('gemini')
                            if api_key:
                                handler = GeminiHandler(api_key=api_key, model=specific_model_name)
                        elif specific_model_name.startswith('claude'):
                            api_key = config.get('api_keys', {}).get('claude')
                            if api_key:
                                handler = ClaudeHandler(api_key=api_key, model=specific_model_name)
                        elif specific_model_name.startswith('gpt'):
                            api_key = config.get('api_keys', {}).get('openai')
                            if api_key:
                                handler = OpenAIHandler(api_key=api_key, model=specific_model_name)
                        else:
                             logger.warning(f"Unknown model type prefix for consultation model: {specific_model_name}. Skipping.")
                             log_step(f"Handler instantiation failed for Step {step_number}", {"model": specific_model_name, "reason": "Unknown model type prefix"})
                             continue

                        if handler and handler.is_ready():
                            consultation_handlers.append(handler)
                            logger.info(f"Successfully instantiated handler: {handler.__class__.__name__} for model {specific_model_name}")
                        elif not api_key:
                             logger.warning(f"API key not found for consultation model: {specific_model_name}. Skipping.")
                             log_step(f"Handler instantiation failed for Step {step_number}", {"model": specific_model_name, "reason": "API key not found"})
                        else:
                             logger.warning(f"Handler for consultation model {specific_model_name} created but is not ready. Skipping.")
                             log_step(f"Handler instantiation failed for Step {step_number}", {"model": specific_model_name, "reason": "Handler not ready (check implementation)"})

                    except Exception as e:
                        logger.error(f"Error instantiating handler for consultation model {specific_model_name}: {e}", exc_info=True)
                        log_step(f"Handler instantiation failed for Step {step_number}", {"model": specific_model_name, "reason": f"Exception: {e}"})


                if not consultation_handlers:
                     log_step(f"Skipping Consultation for Step {step_number}", {"reason": "No valid consultation handlers could be instantiated."})
                     logger.error(f"No valid handlers available for multi-model consultation for step {step_number}. Cannot proceed.")
                     all_step_analysis_results.append({
                         "step_number": step_number,
                         "step_name": step_name,
                         "status": "error",
                         "error_message": "No consultation models could be initialized."
                     })
                     continue

                logger.info(f"Running multi-model consultation for Step {step_number} with {len(consultation_handlers)} handlers.")
                step_consultation_responses: Dict[str, Dict[str, Any]] = await query_manager.run_multi_model_consultation(
                    handlers=consultation_handlers,
                    query=step_prompt,
                    session=session # Pass the session created by async with
                )
                log_step(f"Multi-model consultation complete for Step {step_number}", {"models_responded": list(step_consultation_responses.keys()), "response_summary": {k: v.get('status') for k, v in step_consultation_responses.items()}})

                # Prepare responses for AnalyzerEngine
                analyzer_input_responses: Dict[str, str] = {}
                for model_name, response_data in step_consultation_responses.items():
                    if response_data.get('status') == 'success' and isinstance(response_data.get('content'), str):
                        analyzer_input_responses[model_name] = response_data['content']
                    else:
                        logger.warning(f"Excluding response from {model_name} for step {step_number} analysis due to status '{response_data.get('status')}' or invalid format (content not string?).")
                        log_step(f"Excluding response from Analyzer input for Step {step_number}", {"model": model_name, "status": response_data.get('status'), "reason": "Response not successful or content not a string"})


                # Step 3 & 4: Analyzer AI Evaluation & Segmentation
                log_step(f"Executing Step 3 & 4: Analyzer AI Evaluation & Segmentation for Step {step_number}", {"models_for_analysis": list(analyzer_input_responses.keys())})
                # AnalyzerEngine manages its own session now, call without session
                step_analysis_result = await analyzer_engine.analyze(analyzer_input_responses)
                step_analysis_result["step_number"] = step_number # Add context
                step_analysis_result["step_name"] = step_name
                # Status is now set within analyze method based on success/failure/parsing
                all_step_analysis_results.append(step_analysis_result)
                log_step(f"Analyzer evaluation complete for Step {step_number}", {"status": step_analysis_result["status"], "summary_snippet": step_analysis_result.get('summary', '')[:50] + "..."})
                logger.info(f"--- Completed Step {step_number}: {step_name} ---")


            # Step 5 & 6: Final Architect Analysis & Synthesis
            logger.info("\n--- Starting Final Synthesis ---")
            log_step("Executing Step 5 & 6: Final Architect Analysis & Synthesis", {"num_step_analyses": len(all_step_analysis_results)})
            # SynthesisEngine manages its own session now, call without session
            final_plan_str = await synthesis_engine.synthesize(all_step_analysis_results)
            # Check if synthesis failed before trying to get length
            if isinstance(final_plan_str, str):
                 log_step("Final architect synthesis complete", {"plan_length": len(final_plan_str)})
            else:
                 log_step("Final architect synthesis failed or returned unexpected type", {"result_type": type(final_plan_str).__name__})
            logger.info("--- Final Synthesis Complete ---")


            # --- Output Final Result ---
            print("\n--- Final Synthesized Architecture Plan ---")
            if final_plan_str and not final_plan_str.startswith("Synthesis failed:") and not final_plan_str.startswith("No analysis results"):
                print(final_plan_str)
            else:
                print(f"ERROR: Final synthesis failed or produced no output.")
                print(f"Details: {final_plan_str}") # Print the error message from the synthesizer
                # Optionally print intermediate analysis results for debugging
                print("\nIntermediate Analysis Results:")
                for analysis in all_step_analysis_results:
                     print(f"  Step {analysis.get('step_number', 'N/A')} ({analysis.get('step_name', 'N/A')}): Status - {analysis.get('status', 'N/A')}")
                     if analysis.get('error_message'):
                         print(f"    Error: {analysis.get('error_message')}")

            print("\n--- End of Plan ---")

    except FileNotFoundError as e:
        error_msg = f"\nERROR: Configuration file not found at {CONFIG_PATH}. {e}"
        log_step("Error: Configuration file not found", {"error": str(e)})
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        sys.exit(1)
    except KeyError as e:
        error_msg = f"\nERROR: Missing expected key in configuration or data: {e}"
        log_step("Error: Missing key", {"error": str(e)})
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        sys.exit(1)
    except ImportError as e:
         error_msg = f"\nERROR: Missing required library. Please install dependencies from requirements.txt. Details: {e}"
         log_step("Error: Missing library", {"error": str(e)})
         logger.error(error_msg, exc_info=True)
         print(error_msg)
         sys.exit(1)
    except ValueError as e: # Catch specific value errors like config load failure
         error_msg = f"\nERROR: A configuration or value error occurred: {e}"
         log_step("Error: Value error", {"error": str(e)})
         logger.error(error_msg, exc_info=True)
         print(error_msg)
         sys.exit(1)
    except Exception as e:
        error_msg = f"\nAn unexpected error occurred during the main workflow: {e}"
        tb_str = traceback.format_exc()
        log_step("Error: Unexpected exception", {"error": str(e), "traceback": tb_str})
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        sys.exit(1)
    # Finally block remains outside the async with block
    finally:
        # Write execution log regardless of success or failure
        end_time = datetime.now()
        duration = end_time - start_time
        log_step("Application finished", {"duration": str(duration)})
        logger.info(f"Application finished. Total time: {duration}")
        write_execution_log(execution_log)


# --- Main Execution ---
if __name__ == '__main__':
    asyncio.run(run_cli()) # Use asyncio.run for the async function