import logging
import json # For parsing potential JSON output from analyzer LLM
from typing import Dict, List, Any, Tuple, Optional # Added Optional
import aiohttp # Import aiohttp

# Import ModelHandler for type hinting
from ..models.base import ModelHandler

# Define standard architectural sections
STANDARD_SECTIONS = [
    "Frontend Design",
    "Backend Systems",
    "API Design",
    "Data Management",
    "Deployment & Infrastructure",
    "Security",
    "Scalability & Performance",
    "Monitoring & Logging",
    "Other Considerations"
]

class AnalyzerEngine:
    """
    Analyzes responses from multiple AI models using a dedicated analyzer model
    to identify consensus, summarize findings, and segment architectural recommendations.
    """
    def __init__(self, config: Dict[str, Any], analyzer_handler: Optional[ModelHandler]):
        """
        Initializes the AnalyzerEngine.

        Args:
            config: The application configuration dictionary.
            analyzer_handler: The instantiated model handler for the analyzer model.
        """
        self.config = config
        self.analyzer_handler = analyzer_handler
        logging.info(f"AnalyzerEngine initialized. Analyzer handler: {analyzer_handler.__class__.__name__ if analyzer_handler else 'None'}")
        if not self.analyzer_handler or not self.analyzer_handler.is_ready():
             logging.warning("Analyzer handler is not provided or not ready. Analysis will likely fail.")


    async def _analyze_with_llm(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Uses the configured analyzer LLM to analyze the consultation responses.
        Creates its own session for the API call.

        Args:
            responses: A dictionary of responses from consultation models.

        Returns:
            A dictionary containing the structured analysis from the LLM,
            expected to include 'summary', 'reasoning', and 'segmented_architecture'.
            Returns an error structure if analysis fails.
        """
        if not self.analyzer_handler or not self.analyzer_handler.is_ready():
            logging.error("Analyzer handler is not available or ready.")
            return {
                'status': 'error',
                'summary': "Error: Analyzer model handler not available or not ready.",
                'reasoning': "Cannot perform LLM-based analysis.",
                'segmented_architecture': {section: [] for section in STANDARD_SECTIONS},
                'error_message': "Analyzer model handler not available or not ready."
            }

        # Construct the prompt for the analyzer model
        prompt = "You are an expert software architect acting as an analyzer. You have received the following recommendations for a specific sub-problem from multiple AI consultants.\n\n"
        prompt += "Consultant Responses:\n"
        for model_name, response_text in responses.items():
            prompt += f"--- Response from {model_name} ---\n{response_text}\n\n"

        prompt += "Your task is to analyze these responses and provide a structured analysis in JSON format. The JSON object should have the following keys:\n"
        prompt += "1.  `summary`: A concise summary of the key recommendations and points of agreement/disagreement.\n"
        prompt += "2.  `reasoning`: Explain the rationale behind the summary, highlighting strengths or weaknesses of different proposals if possible.\n"
        prompt += "3.  `segmented_architecture`: A dictionary where keys are standard architectural sections "
        prompt += f"(use ONLY these keys: {', '.join(STANDARD_SECTIONS)}) and values are lists of strings, "
        prompt += "each string being a specific recommendation relevant to that section, extracted or synthesized from the consultant responses.\n"
        prompt += "Ensure all double quotes within string values in the JSON are properly escaped with a backslash (\\\\\").\n\n" # Added instruction to escape double quotes
        prompt += "Provide ONLY the JSON object in your response, without any introductory text or explanations outside the JSON structure. Ensure the JSON is strictly valid and well-formed, with correct punctuation (commas, colons, brackets, braces).\n" # Added emphasis on valid JSON format
        prompt += "JSON Response:\n"

        logging.debug(f"Sending analysis prompt to {self.analyzer_handler.__class__.__name__}...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create a session specifically for this call
                async with aiohttp.ClientSession() as session:
                    analysis_result = await self.analyzer_handler.execute(prompt, session)

                if analysis_result.get('status') == 'success':
                    content = analysis_result.get('content', '{}')
                    logging.debug(f"Raw LLM response content (Attempt {attempt + 1}): {content}") # Added log for raw content with attempt number
                    # Attempt to parse the JSON response
                    try:
                        # Clean potential markdown code fences and strip whitespace
                        cleaned_content = content.strip()
                        if cleaned_content.startswith("```json"):
                            cleaned_content = cleaned_content[7:-3].strip()
                        elif cleaned_content.startswith("```"):
                             cleaned_content = cleaned_content[3:-3].strip()

                        # Attempt to load JSON
                        parsed_json = json.loads(cleaned_content)

                        # Validate structure (basic check)
                        if isinstance(parsed_json, dict) and \
                           'summary' in parsed_json and \
                           'reasoning' in parsed_json and \
                           'segmented_architecture' in parsed_json and \
                           isinstance(parsed_json['segmented_architecture'], dict):
                            # Ensure all standard sections exist in the segmented output
                            for section in STANDARD_SECTIONS:
                                parsed_json['segmented_architecture'].setdefault(section, [])
                            # Add status explicitly for successful analysis
                            parsed_json['status'] = 'success'
                            return parsed_json
                        else:
                            logging.error(f"Analyzer LLM response is not valid JSON or missing required keys (Attempt {attempt + 1}). Content: {content[:500]}...")
                            raise ValueError("Invalid JSON structure or missing keys in analyzer response.")

                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to decode JSON response from analyzer LLM (Attempt {attempt + 1}): {e}. Content: {content[:500]}...")
                        # If JSON decoding fails, modify the prompt and retry
                        if attempt < max_retries - 1:
                            logging.warning(f"JSON decoding failed. Retrying with modified prompt (Attempt {attempt + 2})...")
                            prompt += f"\n\nPrevious attempt failed to decode JSON: {e}. Please provide a valid JSON object."
                            continue # Retry with modified prompt
                        else:
                            raise ValueError(f"Failed to decode JSON response from analyzer LLM after {max_retries} attempts: {e}")
                    except ValueError as e: # Catch validation errors
                         logging.error(f"Invalid structure in JSON response from analyzer LLM (Attempt {attempt + 1}): {e}. Content: {content[:500]}...")
                         raise e # Re-raise the validation error

                else:
                    error_msg = analysis_result.get('error_message', 'Unknown error from analyzer handler.')
                    logging.error(f"Analyzer LLM execution failed (Attempt {attempt + 1}): {error_msg}")
                    raise RuntimeError(f"Analyzer LLM execution failed (Attempt {attempt + 1}): {error_msg}")

            except Exception as e:
                logging.error(f"Error during analyzer LLM call (Attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    logging.warning(f"LLM call failed. Retrying (Attempt {attempt + 2})...")
                    continue # Retry on general exception
                else:
                    return {
                        'status': 'error', # Add status key for errors
                        'summary': f"Error: Failed to get analysis from LLM after {max_retries} attempts - {e}",
                        'reasoning': "Analysis could not be performed due to repeated errors.",
                        'segmented_architecture': {section: [] for section in STANDARD_SECTIONS},
                        'error_message': str(e) # Include error message
                    }

        # If all retries fail
        return {
            'status': 'error',
            'summary': f"Error: Failed to get analysis from LLM after {max_retries} attempts.",
            'reasoning': "Analysis could not be performed due to repeated errors.",
            'segmented_architecture': {section: [] for section in STANDARD_SECTIONS},
            'error_message': f"Failed to get analysis from LLM after {max_retries} attempts."
        }


    async def analyze(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Main analysis method. Uses an analyzer LLM to perform analysis.
        No longer accepts session argument.

        Args:
            responses: A dictionary where keys are model names (e.g., 'geminihandler')
                       and values are their string responses for a subtask.

        Returns:
            A dictionary containing the analysis results:
            {
                'status': str ('success' or 'error'),
                'summary': str,
                'reasoning': str,
                'segmented_architecture': Dict[str, List[str]],
                'error_message': Optional[str]
            }
            Returns an error structure if analysis fails.
        """
        logging.info(f"Starting LLM-based analysis for subtask with responses from: {list(responses.keys())}")

        if not responses:
            logging.warning("No responses provided to analyze.")
            return {
                'status': 'error',
                'summary': "Error: No responses provided.",
                'reasoning': "Cannot perform analysis without input.",
                'segmented_architecture': {section: [] for section in STANDARD_SECTIONS},
                'error_message': "No responses provided to analyze."
            }

        # Call the internal method that creates its own session
        llm_analysis_result = await self._analyze_with_llm(responses)

        # Ensure the result has the expected top-level keys, even if analysis failed
        analysis_result = {
            'status': llm_analysis_result.get('status', 'error'), # Default to error
            'summary': llm_analysis_result.get('summary', "Error: Analysis failed or produced no summary."),
            'reasoning': llm_analysis_result.get('reasoning', "Error: Analysis failed or produced no reasoning."),
            'segmented_architecture': llm_analysis_result.get('segmented_architecture', {section: [] for section in STANDARD_SECTIONS}),
            'error_message': llm_analysis_result.get('error_message') # Include error message if present
        }

        # Ensure segmented_architecture contains all standard sections, even if the LLM missed some
        if isinstance(analysis_result['segmented_architecture'], dict):
             for section in STANDARD_SECTIONS:
                  analysis_result['segmented_architecture'].setdefault(section, [])
        else:
             logging.warning("LLM analysis result for 'segmented_architecture' was not a dict. Resetting.")
             analysis_result['segmented_architecture'] = {section: [] for section in STANDARD_SECTIONS}
             if analysis_result['status'] == 'success': # If it was success but seg arch was bad, mark as error
                  analysis_result['status'] = 'error'
                  analysis_result['error_message'] = (analysis_result.get('error_message', '') + " Segmented architecture was not a dictionary.").strip()


        logging.info("LLM-based analysis complete.")
        logging.debug(f"Analysis Result Snippet: status='{analysis_result['status']}', summary='{analysis_result['summary'][:100]}...', reasoning='{analysis_result['reasoning'][:100]}...', sections={list(analysis_result['segmented_architecture'].keys())}")

        return analysis_result

# Keep the example usage block, but it won't work correctly without
# providing a config and a mock handler during instantiation.
# It's better to test this via the main script execution.
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#     # ... (Example usage would need modification) ...