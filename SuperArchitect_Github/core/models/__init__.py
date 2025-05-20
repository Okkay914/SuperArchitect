# SuperArchitect/core/models/__init__.py
"""
Model Handlers Package

This package contains the base class and specific implementations for interacting
with different AI models via their APIs, using configurations derived from CURL commands.
"""

import logging
from .curl_parser import parse_curl
from .base import ModelHandler
from .openai import OpenAIHandler
from .claude import ClaudeHandler
from .gemini import GeminiHandler

# Configure logging
logger = logging.getLogger(__name__)

# --- Model Handler Factory ---

HANDLER_MAP = {
    'openai': OpenAIHandler,
    'claude': ClaudeHandler,
    'gemini': GeminiHandler,
    # Add other model types here as they are implemented
}

def get_model_handler_for_role(role: str, config: dict) -> ModelHandler | None:
    """
    Factory function to get the configured model handler for a specific role.

    Args:
        role: The role name (e.g., 'summarizing_model', 'final_architect_model')
              as defined in the config.yaml 'models' section.
        config: The loaded configuration dictionary.

    Returns:
        An instantiated ModelHandler or None if configuration is missing or invalid.
    """
    if not config:
        logger.error("Cannot get model handler: Configuration dictionary not provided.")
        return None

    # 1. Get model name for the role from config
    model_config = config.get('models', {}) # Use lowercase 'models' as per main.py load_config
    model_name = model_config.get(role)
    if not model_name:
        logger.error(f"Model name for role '{role}' not found in configuration (MODELS section).")
        return None

    # 2. Get the handler class based on the base model type
    # Extract the base model type (e.g., 'gemini' from 'gemini-2.5-pro-preview-03-25')
    # Extract the base model type (e.g., 'gemini' from 'gemini-2.5-pro-preview-03-25')
    # Handle cases like 'o4-mini' for OpenAI
    derived_base_type = model_name.split('-')[0].lower() if '-' in model_name else model_name.lower()
    
    if derived_base_type == 'o4' or model_name.startswith(('gpt', 'text-', 'ada', 'babbage', 'curie', 'davinci')):
        base_model_type = 'openai'
    elif derived_base_type.startswith('claude'): # Handles cases like 'claude-3-opus-20240229'
        base_model_type = 'claude'
    elif derived_base_type.startswith('gemini'): # Handles cases like 'gemini-1.5-pro-latest'
        base_model_type = 'gemini'
    else:
        base_model_type = derived_base_type

    print(f"DEBUG [models/__init__.py]: For role '{role}', model_name='{model_name}', derived_base_type='{derived_base_type}', final base_model_type='{base_model_type}'")
    
    handler_class = HANDLER_MAP.get(base_model_type)
    if not handler_class:
        error_msg = f"No handler found for base model type '{base_model_type}' derived from '{model_name}' specified for role '{role}'. Available handlers: {list(HANDLER_MAP.keys())}"
        print(f"ERROR [models/__init__.py]: {error_msg}")
        logger.error(error_msg)
        return None

    # 3. Get the API key for the model from config using the base model type
    api_keys_config = config.get('api_keys', {}) # Use lowercase 'api_keys' as per main.py load_config
    print(f"DEBUG [models/__init__.py]: Available API keys: {list(api_keys_config.keys())}")
    
    api_key = api_keys_config.get(base_model_type)
    if not api_key:
        warning_msg = f"API key for model '{model_name}' (role: '{role}') not found in configuration (API_KEYS section) or environment variables. Handler might fail."
        print(f"WARNING [models/__init__.py]: {warning_msg}")
        logger.warning(warning_msg)
        # Proceeding with None key, handler's __init__ should handle this gracefully if possible

    # 4. Instantiate and return the handler
    try:
        logger.info(f"Instantiating {handler_class.__name__} for role '{role}' (model: '{model_name}').")
        # Note: We are not passing model_name or curl_config here,
        # Pass the specific model name from config to the handler constructor
        return handler_class(api_key=api_key, model=model_name)
    except Exception as e:
        logger.error(f"Error instantiating {handler_class.__name__} for role '{role}': {e}")
        return None


__all__ = [
    'parse_curl',
    'ModelHandler',
    'OpenAIHandler',
    'ClaudeHandler',
    'GeminiHandler',
    'get_model_handler_for_role', # Add the factory function
]

# --- Deprecated/Placeholder Factory ---
# def get_model_handler(model_type: str, model_name: str, curl_config: str) -> ModelHandler | None: # Keep old signature for reference if needed
#     """DEPRECATED: Use get_model_handler_for_role instead."""
#     logger.warning("get_model_handler is deprecated. Use get_model_handler_for_role.")
#     # ... (original logic if needed for backward compatibility, but ideally removed)
#     return None