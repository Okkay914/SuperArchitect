import aiohttp
import json
import traceback
import logging
from .base import ModelHandler

# Configure logging
logger = logging.getLogger(__name__)

# Define constants for OpenAI API
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions" # Common endpoint
DEFAULT_MODEL = "gpt-4o-mini" # Use user-specified model

class OpenAIHandler(ModelHandler):
    """Handles interaction with the OpenAI Chat Completions API."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        """
        Initializes the OpenAIHandler.

        Args:
            api_key: The OpenAI API key.
            model: The specific OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        super().__init__(api_key=api_key)
        self.model = model
        self.api_endpoint = OPENAI_API_ENDPOINT
        logger.info(f"OpenAIHandler initialized for model: {self.model}")

    async def execute(self, query: str, session: aiohttp.ClientSession, iteration: int = 1) -> dict:
        """
        Executes the API call to OpenAI.
        """
        if not self.is_ready():
            return {'status': 'error', 'content': None, 'error_message': '[OpenAI] API key not configured.'}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Define the architect persona system prompt
        architect_prompt = (
            "You are an AI assistant acting as a meticulous software architect. "
            "Your goal is to analyze user requests, break them down into logical steps, "
            "create detailed plans, and provide specific, actionable recommendations, "
            "including potential tools or approaches. Respond in a structured manner."
        )

        # Construct the payload according to OpenAI's Chat Completions API
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": architect_prompt}, # Add system prompt
                {"role": "user", "content": query}
            ],
            # Add other parameters like temperature, max_tokens if needed
            # "temperature": 0.7,
            # "max_tokens": 150,
        }

        try:
            logger.debug(f"[OpenAI Iteration {iteration}] Requesting completion for query (first 50 chars): {query[:50]}...")
            async with session.post(self.api_endpoint, headers=headers, json=payload) as response:
                response_text = await response.text()
                response_status = response.status

                if 200 <= response_status < 300:
                    try:
                        data = json.loads(response_text)
                        # --- OpenAI Specific Response Parsing ---
                        content = ""
                        if isinstance(data.get('choices'), list) and len(data['choices']) > 0:
                            choice = data['choices'][0]
                            if isinstance(choice.get('message'), dict):
                                content = choice['message'].get('content', '')
                            elif isinstance(choice.get('delta'), dict): # Handling streaming delta
                                content = choice['delta'].get('content', '')

                        if not content:
                            logger.warning(f"[OpenAI Iteration {iteration}] Could not extract 'content' from choices. Response: {response_text[:200]}...")
                            content = str(data) # Fallback

                        logger.debug(f"[OpenAI Iteration {iteration}] Request successful. Status: {response_status}. Content length: {len(content)}")
                        return {'status': 'success', 'content': content, 'error_message': None}
                    except json.JSONDecodeError:
                         logger.error(f"[OpenAI Iteration {iteration}] Failed to decode JSON response. Status: {response_status}. Response: {response_text[:200]}...")
                         return {'status': 'error', 'content': None, 'error_message': f'[OpenAI] Failed to decode JSON response: {response_text[:200]}...'}
                    except Exception as e: # Catch other parsing errors
                         tb = traceback.format_exc()
                         logger.error(f"[OpenAI Iteration {iteration}] Error parsing response content: {e}. Response: {response_text[:200]}...\n{tb}")
                         return {'status': 'error', 'content': None, 'error_message': f'[OpenAI] Error parsing response content: {e} | Response: {response_text[:200]}...\n{tb}'}
                else:
                    # Handle API errors (non-2xx status codes)
                    error_message = f'[OpenAI] API Error {response_status}: {response_text}'
                    try:
                        error_data = json.loads(response_text)
                        if isinstance(error_data.get('error'), dict):
                            error_message += f" | Type: {error_data['error'].get('type')}, Message: {error_data['error'].get('message')}"
                    except json.JSONDecodeError:
                        pass # Keep the original error message
                    logger.error(f"[OpenAI Iteration {iteration}] API Error. Status: {response_status}. Message: {error_message}")
                    return {'status': 'error', 'content': None, 'error_message': error_message}

        except aiohttp.ClientError as e:
            logger.error(f"[OpenAI Iteration {iteration}] Request failed (aiohttp.ClientError): {e}")
            return {'status': 'error', 'content': None, 'error_message': f'[OpenAI] Request failed (aiohttp.ClientError): {e}'}
        except Exception as e: # Catch unexpected errors
             tb = traceback.format_exc()
             logger.error(f"[OpenAI Iteration {iteration}] Unexpected error during execution: {e}\n{tb}")
             return {'status': 'error', 'content': None, 'error_message': f'[OpenAI] Unexpected error during execution: {e}\n{tb}'}