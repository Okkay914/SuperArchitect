import abc
import aiohttp
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ModelHandler(abc.ABC):
    """
    Abstract base class for AI model handlers.
    Each handler interacts with a specific model API.
    API keys are provided during instantiation via the factory function.
    """
    def __init__(self, api_key: str | None = None):
        """
        Initializes the ModelHandler.

        Args:
            api_key: The API key for the specific service, loaded from config or env vars.
                     Can be None if the key wasn't found, but execution will likely fail.
        """
        self.api_key = api_key
        if not api_key:
             logger.warning(f"{self.__class__.__name__} initialized without an API key. API calls may fail.")

    @abc.abstractmethod
    async def execute(self, query: str, session: aiohttp.ClientSession, iteration: int = 1) -> dict:
        """
        Executes the API call for the given query using the provided aiohttp session.

        Args:
            query: The input query or prompt for the model.
            session: The aiohttp ClientSession for making requests.
            iteration: An optional identifier for the call attempt (e.g., for logging).

        Returns:
            A dictionary containing:
            {'status': 'success'/'error', 'content': <model_response>, 'error_message': <error_details_if_any>}
        """
        pass

    def is_ready(self) -> bool:
        """Checks if the handler has the necessary API key to operate."""
        return self.api_key is not None

    def __repr__(self):
        return f"<{self.__class__.__name__}(ready={self.is_ready()})>"