import asyncio
import aiohttp # Import aiohttp
import logging
from typing import List, Dict, Any
from .models.base import ModelHandler # Import base handler

# Configure logging
logger = logging.getLogger(__name__)

class QueryManager:
    """
    Manages the execution of queries against specific ModelHandlers using
    a provided ClientSession.
    """
    def __init__(self):
        # Initialization if needed
        pass

    async def run_query(self, handler: ModelHandler, query: str, session: aiohttp.ClientSession, iterations: int = 1) -> List[Dict[str, Any]]:
        """
        Runs a query against a single provided model handler using the provided session.

        Args:
            handler: An instantiated and configured ModelHandler.
            query: The query string to send to the model.
            session: The aiohttp client session to use for requests.
            iterations: The number of times to run the query concurrently.

        Returns:
            A list of result dictionaries, one for each iteration.
        """
        if not handler or not handler.is_ready():
            error_msg = f"Handler not provided or not ready (API key missing?). Handler: {handler}"
            logger.error(error_msg)
            return [{'status': 'error', 'model_name': repr(handler), 'iteration': i + 1, 'content': None, 'error_message': error_msg} for i in range(iterations)]

        if not session or session.closed:
             error_msg = "A valid, open aiohttp.ClientSession must be provided."
             logger.error(error_msg)
             return [{'status': 'error', 'model_name': handler.__class__.__name__, 'iteration': i + 1, 'content': None, 'error_message': error_msg} for i in range(iterations)]


        tasks = []
        results = []
        model_name = handler.__class__.__name__

        # Session is now passed in, no need to create one here
        for i in range(iterations):
            # Pass the provided session to the handler's execute method
            task = asyncio.create_task(handler.execute(query, session, iteration=i+1))
            tasks.append({"task": task, "model_name": model_name, "iteration": i + 1})

        gathered_results = await asyncio.gather(*[t["task"] for t in tasks], return_exceptions=True)

        # Process gathered results, adding context back
        for i, result in enumerate(gathered_results):
            task_info = tasks[i]
            # model_name is already known from task_info
            iteration = task_info["iteration"]

            if isinstance(result, Exception):
                error_repr = repr(result)
                logger.error(f"Task execution error for {model_name} iteration {iteration}: {error_repr}")
                results.append({
                    'status': 'error',
                    'model_name': model_name,
                    'iteration': iteration,
                    'content': None,
                    'error_message': f'Task execution error: {error_repr}'
                })
            elif isinstance(result, dict):
                 # Add model_name and iteration to the result dict from the handler
                 result['model_name'] = model_name
                 result['iteration'] = iteration
                 # Ensure all expected keys are present
                 result.setdefault('status', 'error') # Default to error if missing
                 result.setdefault('content', None)
                 result.setdefault('error_message', 'Handler did not return a status.')
                 results.append(result)
            else:
                 # Unexpected result type
                 error_msg = f'Unexpected result type from handler: {type(result)}'
                 logger.error(f"{error_msg} for {model_name} iteration {iteration}")
                 results.append({
                    'status': 'error',
                    'model_name': model_name,
                    'iteration': iteration,
                    'content': None,
                    'error_message': error_msg
                 })

        return results

    async def run_multi_model_consultation(self, handlers: List[ModelHandler], query: str, session: aiohttp.ClientSession) -> Dict[str, Dict[str, Any]]:
       """
       Runs the same query against multiple provided model handlers concurrently, using the provided session.

       Args:
           handlers: A list of instantiated and configured ModelHandler objects.
           query: The query string to send to all models.
           session: The aiohttp client session to use for requests.

       Returns:
           A dictionary where keys are the model names (lowercase class names)
           and values are the result dictionaries from each handler.
       """
       tasks = []
       results_dict = {}
       handler_map = {} # To map task back to handler info

       # Re-instated session check
       if not session or session.closed:
            error_msg = "A valid, open aiohttp.ClientSession must be provided for multi-model consultation."
            logger.error(error_msg)
            # Return errors for all intended handlers
            for handler in handlers:
                 model_name_key = handler.__class__.__name__.lower() if handler else "unknown_handler"
                 results_dict[model_name_key] = {
                     'status': 'error',
                     'model_name': model_name_key, # Use key name here
                     'content': None,
                     'error_message': error_msg
                 }
            return results_dict

       # Session is passed in, no need for 'async with' here
       for handler in handlers:
           if not handler or not handler.is_ready():
               model_name_key = handler.__class__.__name__.lower() if handler else "unknown_handler"
               error_msg = f"Handler {model_name_key} not provided or not ready (API key missing?)."
               logger.error(error_msg)
               results_dict[model_name_key] = {
                   'status': 'error',
                   'model_name': model_name_key,
                   'content': None,
                   'error_message': error_msg
               }
               continue # Skip this handler

           model_name = handler.__class__.__name__
           model_name_key = model_name.lower() # Use lowercase class name as key
           # Pass the provided session to the handler's execute method
           task = asyncio.create_task(handler.execute(query, session))
           # Store task along with its intended key for the results dict
           task_info = {"task": task, "model_key": model_name_key, "model_name": model_name}
           tasks.append(task_info)
           handler_map[task] = task_info # Map task object back to its info

       # Gather results from all tasks that were created
       if not tasks:
            logger.warning("No valid handlers provided for multi-model consultation.")
            return results_dict # Return empty or only errors if no tasks ran

       gathered_results = await asyncio.gather(*[t["task"] for t in tasks], return_exceptions=True)

       # Process gathered results
       for i, result in enumerate(gathered_results):
           # Find the corresponding task info using the order (since gather preserves it)
           original_task = tasks[i]["task"]
           task_info = handler_map[original_task] # Get info using the task object map
           model_key = task_info["model_key"]
           model_name = task_info["model_name"] # Original class name for reporting

           if isinstance(result, Exception):
               error_repr = repr(result)
               logger.error(f"Task execution error for {model_name}: {error_repr}")
               results_dict[model_key] = {
                   'status': 'error',
                   'model_name': model_name, # Report original name
                   'content': None,
                   'error_message': f'Task execution error: {error_repr}'
               }
           elif isinstance(result, dict):
               # Add model_name to the result dict from the handler
               result['model_name'] = model_name # Ensure original name is in the value dict
               # Ensure all expected keys are present
               result.setdefault('status', 'error')
               result.setdefault('content', None)
               result.setdefault('error_message', 'Handler did not return a status.')
               results_dict[model_key] = result
           else:
               # Unexpected result type
               error_msg = f'Unexpected result type from handler {model_name}: {type(result)}'
               logger.error(error_msg)
               results_dict[model_key] = {
                   'status': 'error',
                   'model_name': model_name,
                   'content': None,
                   'error_message': error_msg
               }

       return results_dict

# --- Example Usage (Commented out as before) ---
# ...