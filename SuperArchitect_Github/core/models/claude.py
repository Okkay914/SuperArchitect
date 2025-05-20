import aiohttp
import json
import traceback
import logging
import os  # Import os module to access environment variables
from .base import ModelHandler

# Configure logging
logger = logging.getLogger(__name__)

# Define constants for Anthropic API
ANTHROPIC_API_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-3-7-sonnet-20250219" # Use user-specified model
ANTHROPIC_VERSION = "2023-06-01" # Required header

class ClaudeHandler(ModelHandler):
    """Handles interaction with the Anthropic Messages API (Claude)."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        """
        Initializes the ClaudeHandler.

        Args:
            api_key: The Anthropic API key. If None, attempts to read from the
                     ANTHROPIC_API_KEY environment variable.
            model: The specific Claude model to use (e.g., 'claude-3-opus-20240229').
        """
        # Prioritize explicitly passed API key, otherwise check environment variable
        effective_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        super().__init__(api_key=effective_api_key)
        self.model = model
        self.api_endpoint = ANTHROPIC_API_ENDPOINT
        logger.info(f"ClaudeHandler initialized for model: {self.model}")

    async def execute(self, query: str, session: aiohttp.ClientSession, iteration: int = 1) -> dict:
        """
        Executes the API call to Anthropic (Claude).
        """
        if not self.is_ready():
            return {'status': 'error', 'content': None, 'error_message': '[Claude] API key not configured.'}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

        # Define the architect persona system prompt
        architect_prompt = (
            "You are an AI assistant acting as a meticulous software architect. "
            "Your goal is to analyze user requests, break them down into logical steps, "
            "create detailed plans, and provide specific, actionable recommendations, "
            "including potential tools or approaches. Respond in a structured manner."
        )

        # Construct the payload according to Anthropic's Messages API
        payload = {
            "model": self.model,
            "system": architect_prompt, # Add the system prompt here
            "messages": [
                {"role": "user", "content": query}
            ],
            "max_tokens": 8000, # Increased further to prevent truncation for detailed JSON
            # Add other parameters like temperature if needed
            # "temperature": 0.7,
        }

        try:
            logger.debug(f"[Claude Iteration {iteration}] Requesting completion for query (first 50 chars): {query[:50]}...")
            async with session.post(self.api_endpoint, headers=headers, json=payload) as response:
                response_text = await response.text()
                logger.debug(f"[Claude Iteration {iteration}] Raw response text: {response_text}")
                response_status = response.status

                if 200 <= response_status < 300:
                    try:
                        data = json.loads(response_text)
                        # --- Claude Specific Response Parsing ---
                        content = ""
                        if isinstance(data.get('content'), list) and len(data['content']) > 0:
                            first_block = data['content'][0]
                            # Check if the first block is text content
                            if isinstance(first_block, dict) and first_block.get('type') == 'text':
                                content = first_block.get('text', '')

                        if not content:
                             logger.warning(f"[Claude Iteration {iteration}] Could not extract text content from response. Response: {response_text[:200]}...")
                             content = str(data) # Fallback

                        logger.debug(f"[Claude Iteration {iteration}] Request successful. Status: {response_status}. Content length: {len(content)}")
                        return {'status': 'success', 'content': content, 'error_message': None}
                    except json.JSONDecodeError:
                         logger.error(f"[Claude Iteration {iteration}] Failed to decode JSON response. Status: {response_status}. Raw Response: {response_text}")
                         return {'status': 'error', 'content': None, 'error_message': f'[Claude] Failed to decode JSON response. Raw Response: {response_text}'}
                    except Exception as e: # Catch other parsing errors
                         tb = traceback.format_exc()
                         logger.error(f"[Claude Iteration {iteration}] Error parsing response content: {e}. Raw Response: {response_text}\n{tb}")
                         return {'status': 'error', 'content': None, 'error_message': f'[Claude] Error parsing response content: {e} | Raw Response: {response_text}\n{tb}'}
                else:
                    # Handle API errors (non-2xx status codes)
                    error_message = f'[Claude] API Error {response_status}: {response_text}'
                    try:
                        error_data = json.loads(response_text)
                        if isinstance(error_data.get('error'), dict):
                            error_message += f" | Type: {error_data['error'].get('type')}, Message: {error_data['error'].get('message')}"
                    except json.JSONDecodeError:
                        pass # Keep the original error message
                    logger.error(f"[Claude Iteration {iteration}] API Error. Status: {response_status}. Message: {error_message}")
                    return {'status': 'error', 'content': None, 'error_message': error_message}

        except aiohttp.ClientError as e:
            logger.error(f"[Claude Iteration {iteration}] Request failed (aiohttp.ClientError): {e}")
            return {'status': 'error', 'content': None, 'error_message': f'[Claude] Request failed (aiohttp.ClientError): {e}'}
        except Exception as e: # Catch unexpected errors
             tb = traceback.format_exc()
             logger.error(f"[Claude Iteration {iteration}] Unexpected error during execution: {e}\n{tb}")
             return {'status': 'error', 'content': None, 'error_message': f'[Claude] Unexpected error during execution: {e}\n{tb}'}