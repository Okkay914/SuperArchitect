import logging
import json # For potential future structured output, though current plan is text
from typing import List, Dict, Any, Optional # Added Optional
import aiohttp # Import aiohttp

# Import ModelHandler for type hinting
from ..models.base import ModelHandler

logger = logging.getLogger(__name__)

class SynthesisEngine:
    """
    Synthesizes analysis results from multiple architectural steps into a final,
    cohesive architectural plan using a dedicated synthesis model.
    """
    def __init__(self, config: Dict[str, Any], synthesis_handler: Optional[ModelHandler]):
        """
        Initializes the SynthesisEngine.

        Args:
            config: The application configuration dictionary.
            synthesis_handler: The instantiated model handler for the synthesis model.
        """
        self.config = config
        self.synthesis_handler = synthesis_handler
        logging.info(f"SynthesisEngine initialized. Synthesis handler: {synthesis_handler.__class__.__name__ if synthesis_handler else 'None'}")
        if not self.synthesis_handler or not self.synthesis_handler.is_ready():
             logging.warning("Synthesis handler is not provided or not ready. Synthesis will likely fail.")


    async def _synthesize_with_llm(self, all_step_analysis_results: list[dict]) -> str:
        """
        Uses the configured synthesis LLM to generate the final architectural plan.
        Creates its own session for the API call.

        Args:
            all_step_analysis_results: A list of analysis result dictionaries from AnalyzerEngine.

        Returns:
            A string containing the final architectural plan, or an error message.
        """
        if not self.synthesis_handler or not self.synthesis_handler.is_ready():
            logging.error("Synthesis handler is not available or ready.")
            return "Synthesis failed: Synthesis model handler not available or not ready."

        # Construct the prompt for the synthesis model
        prompt = "You are an expert software architect responsible for synthesizing a final, cohesive architectural plan.\n"
        prompt += "You have received structured analysis for several sub-problems or steps of the overall architecture design.\n\n"
        prompt += "Analysis Results per Step:\n"
        for i, result in enumerate(all_step_analysis_results):
            step_name = result.get('step_name', f"Step {i+1}")
            summary = result.get('summary', 'N/A')
            reasoning = result.get('reasoning', 'N/A')
            segmented_architecture = result.get('segmented_architecture', {})

            prompt += f"--- Analysis for: {step_name} ---\n"
            prompt += f"Summary: {summary}\n"
            prompt += f"Reasoning: {reasoning}\n"
            prompt += "Segmented Recommendations:\n"
            if isinstance(segmented_architecture, dict):
                 for section, recommendations in segmented_architecture.items():
                      if recommendations: # Only include sections with recommendations
                           prompt += f"  - {section}:\n"
                           for rec in recommendations:
                                prompt += f"    * {rec}\n"
            else:
                 prompt += "  (No valid segmented architecture provided)\n"
            prompt += "\n"

        prompt += "Your task is to synthesize these analyses into a single, coherent, and well-structured final architectural plan document.\n"
        prompt += "The plan should:\n"
        prompt += "- Integrate the findings from each step.\n"
        prompt += "- Resolve any conflicts or inconsistencies between the step analyses.\n"
        prompt += "- Provide clear recommendations for each major architectural area.\n"
        prompt += "- Explain the rationale behind the final choices.\n"
        prompt += "- Be formatted clearly using Markdown.\n\n"
        prompt += "Generate the final architectural plan document now:\n"

        logging.debug(f"Sending synthesis prompt to {self.synthesis_handler.__class__.__name__}...")

        try:
            # Create a session specifically for this call
            async with aiohttp.ClientSession() as session:
                synthesis_result = await self.synthesis_handler.execute(prompt, session)

            if synthesis_result.get('status') == 'success':
                final_plan = synthesis_result.get('content')
                if isinstance(final_plan, str) and final_plan.strip():
                    logging.info("Synthesis LLM call successful.")
                    return final_plan
                else:
                    logging.error("Synthesis LLM returned success status but no valid content.")
                    return "Synthesis failed: LLM returned no content."
            else:
                error_msg = synthesis_result.get('error_message', 'Unknown error from synthesis handler.')
                logging.error(f"Synthesis LLM execution failed: {error_msg}")
                return f"Synthesis failed: {error_msg}"

        except Exception as e:
            logging.error(f"Error during synthesis LLM call: {e}", exc_info=True)
            return f"Synthesis failed: An exception occurred - {e}"


    async def synthesize(self, all_step_analysis_results: list[dict]) -> str:
        """
        Orchestrates the synthesis process using the synthesis LLM.
        No longer accepts session argument.

        Args:
            all_step_analysis_results: A list of dictionaries, where each dictionary
                                       is the output of AnalyzerEngine.analyze for one step.

        Returns:
            A string representing the final synthesized architectural plan.
        """
        logger.info("Starting final architectural synthesis using LLM.")
        if not all_step_analysis_results:
            logger.warning("No analysis results provided for synthesis.")
            return "Synthesis failed: No analysis results were provided."

        # Filter out steps that resulted in errors during analysis
        successful_analyses = [
             result for result in all_step_analysis_results
             if result.get('status') == 'success' # Check status directly
        ]

        if not successful_analyses:
             logger.error("No successful analysis results available to synthesize.")
             # Optionally, summarize the errors from the input
             error_summary = "Synthesis failed: No successful analysis results were available.\nErrors encountered:\n"
             for i, result in enumerate(all_step_analysis_results):
                  step_name = result.get('step_name', f"Step {i+1}")
                  error_msg = result.get('error_message', result.get('summary', 'Unknown error')) # Get specific error message if available
                  error_summary += f"- {step_name}: {error_msg}\n"
             return error_summary


        # Call the internal method that creates its own session
        final_plan = await self._synthesize_with_llm(successful_analyses)

        logger.info("Final architectural synthesis complete.")
        return final_plan

# Remove or update the old example usage block as it's based on placeholder logic
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#     # ... (Mock results need updating if kept) ...
#     synthesizer = SynthesisEngine() # Needs config and handler now
#     final_plan = synthesizer.synthesize(mock_results)
#     print("\n--- Final Synthesized Plan ---")
#     print(final_plan)