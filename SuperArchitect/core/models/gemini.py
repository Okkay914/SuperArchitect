import google.generativeai as genai
import traceback
import logging
import os  # Added import
from .base import ModelHandler

# Configure logging
logger = logging.getLogger(__name__)

class GeminiHandler(ModelHandler):
    """Handles interaction with the Google Gemini API using google-genai."""

    def __init__(self, model: str, api_key: str | None = None):
        """
        Initializes the GeminiHandler.

        Args:
            model: The specific Gemini model to use (e.g., 'gemini-pro').
            api_key: The Google AI Studio API key.
        """
        super().__init__(api_key=api_key) # Initialize base class with potentially None key first
        self.model = model
        self.client = None

        print(f"DEBUG [gemini.py]: Initializing GeminiHandler with model='{model}', api_key_explicitly_provided={api_key is not None}")

        # Prioritize explicitly passed API key, otherwise check environment variable
        if not self.api_key:
            print(f"DEBUG [gemini.py]: No API key passed explicitly, checking environment variable GOOGLE_API_KEY...")
            self.api_key = os.environ.get("GOOGLE_API_KEY")
            if self.api_key:
                print(f"DEBUG [gemini.py]: Found API key in environment variable GOOGLE_API_KEY.")
            else:
                print(f"ERROR [gemini.py]: GOOGLE_API_KEY environment variable not set and no key passed explicitly.")
                # Raise error immediately if no key is found anywhere
                raise ValueError("Gemini API key not provided via constructor or GOOGLE_API_KEY environment variable.")
        else:
             print(f"DEBUG [gemini.py]: Using explicitly passed API key.")


        # Now, self.api_key is guaranteed to be set if we reach here
        try:
            print(f"DEBUG [gemini.py]: Configuring genai with API key and creating client for model: {self.model}")
            genai.configure(api_key=self.api_key) # Configure using the determined API key
            self.client = genai.GenerativeModel(model_name=self.model)
            logger.info(f"GeminiHandler initialized for model: {self.model}")
            print(f"DEBUG [gemini.py]: Successfully initialized client for model: {self.model}")
        except Exception as e:
            error_msg = f"Failed to initialize Google Gemini client: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            print(f"ERROR [gemini.py]: {error_msg}")
            self.client = None
        # Removed the original 'else' block as the logic is now handled above,
        # and an error is raised if no key is found.


    async def execute(self, query: str, session=None, iteration: int = 1) -> dict:
        """
        Executes the API call to Google Gemini using google-genai.
        """
        print(f"DEBUG [gemini.py]: execute() called with query length: {len(query)}, iteration: {iteration}")
        
        if not self.is_ready() or not self.client:
            error_msg = '[Gemini] API key not configured or client failed to initialize.'
            print(f"ERROR [gemini.py]: {error_msg}")
            return {'status': 'error', 'content': None, 'error_message': error_msg}

        try:
            print(f"DEBUG [gemini.py]: Preparing to send request to Gemini API for iteration {iteration}")
            logger.debug(f"[Gemini Iteration {iteration}] Requesting completion for query (first 50 chars): {query[:50]}...")

            # Define the architect persona system prompt
            architect_prompt = (
                "You are an AI assistant acting as a meticulous software architect. "
                "Your goal is to analyze user requests, break them down into logical steps, "
                "create detailed plans, and provide specific, actionable recommendations, "
                "including potential tools or approaches. Respond in a structured manner.\n\n"
                "User Query:\n"
            )
            full_query = architect_prompt + query
            print(f"DEBUG [gemini.py]: Created full query with architect prompt, total length: {len(full_query)}")

            # Use the google-genai client to generate content
            print(f"DEBUG [gemini.py]: Calling generate_content_async with model: {self.model}")
            try:
                response = await self.client.generate_content_async(full_query)
                print(f"DEBUG [gemini.py]: Response received from Gemini API")
            except Exception as api_error:
                error_detail = f"API call failed: {str(api_error)}\n{traceback.format_exc()}"
                print(f"ERROR [gemini.py]: {error_detail}")
                return {'status': 'error', 'content': None, 'error_message': f'[Gemini] {error_detail}'}

            # --- google-genai Specific Response Parsing ---
            content = ""
            if response and hasattr(response, 'text'):
                content = response.text
            elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 logger.warning(f"[Gemini Iteration {iteration}] Request blocked by API. Reason: {block_reason}. Response: {response}")
                 return {'status': 'error', 'content': None, 'error_message': f'[Gemini] Request blocked by API. Reason: {block_reason}.'}
            else:
                 logger.warning(f"[Gemini Iteration {iteration}] Could not extract text content from response. Response: {response}")
                 content = str(response) # Fallback

            if content:
                logger.debug(f"[Gemini Iteration {iteration}] Request successful. Content length: {len(content)}")
                return {'status': 'success', 'content': content, 'error_message': None}
            else:
                 logger.error(f"[Gemini Iteration {iteration}] Received empty content from API. Response: {response}")
                 return {'status': 'error', 'content': None, 'error_message': f'[Gemini] Received empty content from API. Response: {response}'}


        except Exception as e: # Catch unexpected errors from google-genai
             tb = traceback.format_exc()
             logger.error(f"[Gemini Iteration {iteration}] Unexpected error during execution: {e}\n{tb}")
             return {'status': 'error', 'content': None, 'error_message': f'[Gemini] Unexpected error during execution: {e}\n{tb}'}