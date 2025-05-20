import os
import io
import traceback
import logging
from typing import List, Union, Optional, Dict, Any

from PIL import Image

# Attempt to import google-genai SDK
try:
    import google.generativeai as genai
    import google.generativeai.types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    class genai: # type: ignore
        @staticmethod
        def configure(api_key): pass
        class GenerativeModel: pass 
    class genai_types: # type: ignore
        class GenerationConfig: pass


from .base import ModelHandler

logger = logging.getLogger(__name__)

class GeminiHandler(ModelHandler):
    """
    Handles interaction with Google Gemini models via the google-generativeai SDK.
    """

    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 system_instruction_text: Optional[str] = None,
                 default_generation_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the GeminiHandler.

        Args:
            model: The specific Gemini model to use (e.g., 'gemini-1.5-pro-preview-0409', 'gemini-2.5-pro-preview-05-06').
            api_key: API key for google-generativeai SDK.
            system_instruction_text: System-level instructions for the model.
            default_generation_config: Default parameters for generation (e.g., temperature, max_output_tokens).
        """
        super().__init__(api_key=api_key)

        self.model = model
        self.system_instruction_text = system_instruction_text
        self.default_generation_config = default_generation_config if default_generation_config else {}

        self.client: Optional[genai.GenerativeModel] = None # type: ignore
        self.client_type: Optional[str] = None  # Will be "genai" if successful

        effective_api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if effective_api_key:
            if GENAI_AVAILABLE:
                logger.info(f"Attempting to initialize google-genai client for model='{self.model}'")
                print(f"DEBUG [gemini.py]: Initializing google-genai client for model='{self.model}'")
                try:
                    genai.configure(api_key=effective_api_key) # type: ignore
                    self.client = genai.GenerativeModel(model_name=self.model) # type: ignore
                    self.client_type = "genai"
                    logger.info(f"GeminiHandler initialized with google-genai for model: {self.model}")
                    print(f"DEBUG [gemini.py]: Successfully initialized google-genai client for model: {self.model}")
                except Exception as e:
                    error_msg = f"Failed to initialize google-genai client: {e}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    print(f"ERROR [gemini.py]: {error_msg}")
            else:
                logger.warning("google-generativeai SDK not available. Cannot initialize genai client.")
                print("WARNING [gemini.py]: google-generativeai SDK not available.")
        else:
             logger.info("No API key for google-genai found (neither explicit nor GOOGLE_API_KEY env var).")
             print("DEBUG [gemini.py]: No API key for google-genai.")


        if not self.client:
            final_error_msg = "GeminiHandler failed to initialize: No valid API key or google-generativeai SDK not available."
            logger.error(final_error_msg)
            print(f"ERROR [gemini.py]: {final_error_msg}")
            raise ValueError(final_error_msg)

    def is_ready(self) -> bool:
        """Checks if the client is initialized and ready."""
        return self.client is not None

    async def execute(self,
                      query_parts: List[Union[str, Image.Image]],
                      session=None, # Kept for interface compatibility, currently unused
                      iteration: int = 1,
                      **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the request to the Gemini model.

        Args:
            query_parts: A list of parts for the query, can be strings or PIL.Image.Image objects.
            session: (Optional) The client session.
            iteration: (Optional) The iteration number for logging.
            **kwargs: Additional generation parameters to override defaults.

        Returns:
            A dictionary with 'status', 'content', and 'error_message'.
        """
        logger.debug(f"[Gemini Iteration {iteration}] execute() called with {len(query_parts)} parts. Client type: {self.client_type}")

        if not self.is_ready() or not self.client:
            error_msg = f'[Gemini] Client not properly initialized. Client type: {self.client_type}.'
            logger.error(error_msg)
            print(f"ERROR [gemini.py]: {error_msg}")
            return {'status': 'error', 'content': None, 'error_message': error_msg}

        try:
            if self.client_type == "genai" and GENAI_AVAILABLE:
                print(f"DEBUG [gemini.py]: Using google-genai client for model: {self.model}")
                genai_api_parts: List[Any] = []
                if self.system_instruction_text:
                    genai_api_parts.append(self.system_instruction_text)
                
                for part_data in query_parts:
                    if isinstance(part_data, (str, Image.Image)): # genai handles PIL Images directly
                        genai_api_parts.append(part_data)
                    # genai.Part is different, so we don't explicitly handle it unless it's from genai itself
                    else:
                        logger.warning(f"Unsupported part type for google-genai: {type(part_data)}. Skipping.")
                        print(f"WARNING [gemini.py]: Unsupported part type {type(part_data)} for google-genai. Skipping.")

                if not genai_api_parts:
                     return {'status': 'error', 'content': None, 'error_message': '[Gemini genai] No valid parts to send after preparation.'}

                genai_generation_config_obj = None
                temp_config = self.default_generation_config.copy()
                temp_config.update(kwargs)
                
                known_genai_keys = ["temperature", "top_p", "top_k", "candidate_count", "max_output_tokens", "stop_sequences"]
                genai_compatible_config_dict = {
                    k: v for k, v in temp_config.items() if k in known_genai_keys and v is not None
                }
                if genai_compatible_config_dict:
                    try:
                        genai_generation_config_obj = genai_types.GenerationConfig(**genai_compatible_config_dict) # type: ignore
                    except Exception as e_conf:
                        logger.error(f"[Gemini genai] Failed to create genai.types.GenerationConfig: {e_conf}. Proceeding without specific config.")
                        print(f"ERROR [gemini.py]: Failed to create genai.types.GenerationConfig: {e_conf}")

                logger.debug(f"[Gemini genai Iteration {iteration}] Sending request. Parts: {len(genai_api_parts)}. Config: {genai_compatible_config_dict if genai_compatible_config_dict else 'default'}")
                print(f"DEBUG [gemini.py]: Calling genai generate_content_async. Model: {self.model}")
                
                response = await self.client.generate_content_async( # type: ignore
                    genai_api_parts,
                    generation_config=genai_generation_config_obj
                )
                print(f"DEBUG [gemini.py]: Response received from genai API")

                content = ""
                if response and hasattr(response, 'text'):
                    content = response.text
                elif response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: # type: ignore
                    block_reason = response.prompt_feedback.block_reason # type: ignore
                    logger.warning(f"[Gemini genai Iteration {iteration}] Request blocked. Reason: {block_reason}. Response: {response}")
                    return {'status': 'error', 'content': None, 'error_message': f'[Gemini genai] Request blocked. Reason: {block_reason}.'}
                else:
                    logger.warning(f"[Gemini genai Iteration {iteration}] Could not extract text from response. Fallback: {str(response)}")
                    content = str(response) 

                if content:
                    logger.debug(f"[Gemini genai Iteration {iteration}] Success. Content length: {len(content)}")
                    return {'status': 'success', 'content': content, 'error_message': None}
                else:
                    logger.error(f"[Gemini genai Iteration {iteration}] Empty content. Response: {response}")
                    return {'status': 'error', 'content': None, 'error_message': f'[Gemini genai] Received empty content. Response: {response}'}
            
            else: # Should not happen if __init__ is correct and SDKs are available as expected
                error_msg = f"[Gemini] Client type '{self.client_type}' is not recognized or its SDK is unavailable."
                logger.error(error_msg)
                print(f"ERROR [gemini.py]: {error_msg}")
                return {'status': 'error', 'content': None, 'error_message': error_msg}

        except Exception as e:
            tb = traceback.format_exc()
            error_src = self.client_type if self.client_type else "UnknownClient"
            logger.error(f"[Gemini {error_src} Iteration {iteration}] Unexpected error during execution: {e}\n{tb}")
            print(f"ERROR [gemini.py]: Unexpected error for {error_src} client: {e}\n{tb}")
            return {'status': 'error', 'content': None, 'error_message': f'[Gemini {error_src}] Unexpected error: {e}'}