import logging
import json # For potential future structured output, though current plan is text
from typing import List, Dict, Any, Optional
import aiohttp

# Import ModelHandler for type hinting
from ..models.base import ModelHandler

logger = logging.getLogger(__name__)


def format_json_to_markdown(analysis_json: Dict[str, Any]) -> str:
    """
    Formats a structured JSON analysis into a markdown string.
    """
    markdown_parts = []
    # Define a preferred order for common sections to ensure consistency
    preferred_order = [
        "Summary/Overview", "Executive Summary",
        "Key Findings", "Key Insights",
        "Key Considerations", "Key Considerations/Factors", "Important Factors",
        "Analysis & Reasoning", "Rationale", "Detailed Analysis",
        "Recommended Approach", "Recommended Approach/Design", "Proposed Solution",
        "Implementation Details", "Technical Specifications",
        "Potential Challenges", "Potential Challenges/Risks", "Risks and Mitigations",
        "Mitigation Strategies",
        "Actionable Next Steps", "Next Steps",
        "Success Metrics", "KPIs",
        "Conclusion", "Final Thoughts"
    ]

    processed_keys = set()

    # Add known keys first, in preferred order
    for key_variant in preferred_order:
        for actual_key in analysis_json:
            if actual_key.lower() == key_variant.lower() and actual_key not in processed_keys:
                content = analysis_json[actual_key]
                if content: # Ensure content is not None or empty
                    markdown_parts.append(f"### {actual_key}\n")
                    if isinstance(content, str):
                        if content.strip() and content.strip() != f"No content provided for {actual_key}.":
                            markdown_parts.append(f"{content.strip()}\n")
                    elif isinstance(content, list):
                        for item in content:
                            markdown_parts.append(f"- {item}\n")
                    elif isinstance(content, dict):
                        for sub_key, sub_value in content.items():
                            markdown_parts.append(f"- **{sub_key}:** {sub_value}\n")
                    markdown_parts.append("\n") # Add a blank line after the section
                processed_keys.add(actual_key)
                break # Move to next preferred_order key

    # Add any other keys not in preferred_order and not yet processed
    for key, content in analysis_json.items():
        if key not in processed_keys and content:
            markdown_parts.append(f"### {key}\n")
            if isinstance(content, str):
                 if content.strip() and content.strip() != f"No content provided for {key}.":
                    markdown_parts.append(f"{content.strip()}\n")
            elif isinstance(content, list):
                for item in content:
                    markdown_parts.append(f"- {item}\n")
            elif isinstance(content, dict):
                for sub_key, sub_value in content.items():
                    markdown_parts.append(f"- **{sub_key}:** {sub_value}\n")
            markdown_parts.append("\n")
            processed_keys.add(key)

    return "".join(markdown_parts)


def append_substep_analysis_to_markdown(substep_title: str, analysis_json: Dict[str, Any], output_filepath: str) -> None:
    """
    Formats a substep's analysis JSON into markdown and appends it to the output file.
    Creates the file if it doesn't exist (though header should do that first).
    Args:
        substep_title: The title of the substep (e.g., "Define Lead Generation Objectives").
        analysis_json: The structured JSON analysis for this substep.
        output_filepath: Path to the markdown output file.
    """
    logger.info(f"Appending analysis for substep '{substep_title}' to {output_filepath}")
    try:
        # Slugify title for anchor link, consistent with TOC generation
        anchor = substep_title.lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "").replace(":", "").replace("'", "")
        markdown_content = f"<a name=\"{anchor}\"></a>\n" # HTML anchor for TOC
        markdown_content += f"## {substep_title}\n\n"
        
        formatted_analysis = format_json_to_markdown(analysis_json)
        if not formatted_analysis.strip():
            formatted_analysis = "No detailed analysis content was generated for this substep.\n"
            
        markdown_content += formatted_analysis
        markdown_content += "\n---\n\n"  # Separator between substeps

        with open(output_filepath, 'a', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.debug(f"Successfully appended analysis for '{substep_title}'.")
    except Exception as e:
        logger.error(f"Error appending substep analysis for '{substep_title}' to {output_filepath}: {e}", exc_info=True)


class SynthesisEngine:
    """
    Manages the generation of the header (introduction, TOC) and conclusion
    for the final architectural plan. Substep content is appended incrementally
    by an external function.
    """
    def __init__(self, config: Dict[str, Any], synthesis_handler: Optional[ModelHandler]):
        """
        Initializes the SynthesisEngine.
        Args:
            config: The application configuration dictionary.
            synthesis_handler: The instantiated model handler for synthesis tasks (intro/conclusion).
        """
        self.config = config
        self.synthesis_handler = synthesis_handler
        logging.info(f"SynthesisEngine initialized. Synthesis handler: {synthesis_handler.__class__.__name__ if synthesis_handler else 'None'}")
        if not self.synthesis_handler or not self.synthesis_handler.is_ready():
             logging.warning("Synthesis handler is not provided or not ready. LLM-based intro/conclusion may not be available.")

    def is_ready(self) -> bool:
        """
        Checks if the SynthesisEngine is ready to process requests.
        This mainly verifies if the synthesis handler is available and ready.
        
        Returns:
            bool: True if the engine is ready, False otherwise.
        """
        return self.synthesis_handler is not None and self.synthesis_handler.is_ready()

    def _generate_toc_markdown(self, substep_titles: List[str]) -> str:
        """Generates a markdown Table of Contents from substep titles."""
        if not substep_titles:
            return ""
        toc = "## Table of Contents\n\n"
        for i, title in enumerate(substep_titles):
            # Slugify title for anchor links, consistent with append_substep_analysis_to_markdown
            anchor = title.lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "").replace(":", "").replace("'", "")
            toc += f"{i+1}. [{title}](#{anchor})\n"
        toc += "\n---\n\n"
        return toc

    async def write_document_header(self, output_filepath: str, main_query: str, substep_titles: List[str]) -> None:
        """
        Writes an overall introduction and table of contents to the beginning of the output file.
        This should be called ONCE before any substeps are appended.
        It creates/overwrites the output file.
        """
        logger.info(f"Writing document header for query '{main_query}' to {output_filepath}")
        header_content = f"# Architectural Plan for: {main_query}\n\n"
        introduction_text = "This document outlines the architectural plan generated based on the provided query and subsequent analysis of its constituent substeps.\n\n"

        if self.synthesis_handler and self.synthesis_handler.is_ready():
            prompt = (
                f"You are an expert technical writer. Generate a concise and professional introduction "
                f"for an architectural plan document titled 'Architectural Plan for: {main_query}'.\n"
                "The document will detail the following major sections (substeps):\n"
            )
            for title in substep_titles:
                prompt += f"- {title}\n"
            prompt += (
                "\nThe introduction should briefly state the purpose of the document. "
                "Do not generate the content for the sections themselves, only the introduction paragraph(s).\n"
                "Following the introduction, a Table of Contents will be programmatically generated. "
                "Your output should ONLY be the introduction text, formatted in Markdown."
            )
            logger.debug(f"Sending introduction prompt to {self.synthesis_handler.__class__.__name__}...")
            try:
                async with aiohttp.ClientSession() as session:
                    intro_result = await self.synthesis_handler.execute(prompt, session)
                if intro_result.get('status') == 'success' and intro_result.get('content'):
                    introduction_text = intro_result['content'].strip() + "\n\n"
                    logger.info("Introduction successfully generated by LLM.")
                else:
                    logger.warning(f"Failed to generate introduction via LLM: {intro_result.get('error_message', 'No content')}. Using generic one.")
            except Exception as e:
                logger.error(f"Error generating introduction via LLM: {e}. Using generic one.", exc_info=True)
        else:
            logger.info("Synthesis handler not available or ready. Using generic introduction.")

        header_content += introduction_text
        header_content += self._generate_toc_markdown(substep_titles)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f: # 'w' to overwrite/create
                f.write(header_content)
            logger.info(f"Document header (intro + TOC) written to {output_filepath}")
        except Exception as e:
            logger.error(f"Error writing document header to {output_filepath}: {e}", exc_info=True)

    async def write_document_conclusion(self, output_filepath: str, main_query: str) -> None:
        """
        Appends an overall conclusion to the output file.
        This should be called ONCE after all substeps have been appended.
        """
        logger.info(f"Writing document conclusion for query '{main_query}' to {output_filepath}")
        conclusion_text = (
            "\n## Conclusion\n\nThis document provides a detailed architectural plan based on the analysis of various substeps "
            "related to the initial query. Further refinements and detailed design work may be necessary based on evolving requirements.\n"
        )

        if self.synthesis_handler and self.synthesis_handler.is_ready():
            prompt = (
                f"You are an expert technical writer. Generate a concise and professional concluding summary "
                f"for an architectural plan document that addressed the main query: '{main_query}'.\n"
                "The document has already detailed the analysis and recommendations for various substeps.\n"
                "The conclusion should summarize the purpose of the document and perhaps suggest general next steps or "
                "considerations without reiterating specific details from the substeps already covered.\n"
                "Format the conclusion in Markdown. Your output should ONLY be the conclusion section text, "
                "starting with a '## Conclusion' heading if appropriate, or just the text to be placed under such a heading."
            )
            logger.debug(f"Sending conclusion prompt to {self.synthesis_handler.__class__.__name__}...")
            try:
                async with aiohttp.ClientSession() as session:
                    conclusion_result = await self.synthesis_handler.execute(prompt, session)
                if conclusion_result.get('status') == 'success' and conclusion_result.get('content'):
                    generated_conclusion = conclusion_result['content'].strip()
                    if not generated_conclusion.lower().lstrip().startswith("## conclusion"):
                         conclusion_text = "\n## Conclusion\n\n" + generated_conclusion + "\n"
                    else:
                         conclusion_text = "\n" + generated_conclusion + "\n"
                    logger.info("Conclusion successfully generated by LLM.")
                else:
                    logger.warning(f"Failed to generate conclusion via LLM: {conclusion_result.get('error_message', 'No content')}. Using generic one.")
            except Exception as e:
                logger.error(f"Error generating conclusion via LLM: {e}. Using generic one.", exc_info=True)
        else:
            logger.info("Synthesis handler not available or ready. Using generic conclusion.")

        try:
            with open(output_filepath, 'a', encoding='utf-8') as f: # 'a' to append
                f.write(conclusion_text)
            logger.info(f"Document conclusion appended to {output_filepath}")
        except Exception as e:
            logger.error(f"Error appending document conclusion to {output_filepath}: {e}", exc_info=True)

    async def append_error_to_markdown(self, output_filepath: str, substep_title: str, error_message: str) -> None:
        """
        Appends a formatted error message for a specific substep to the output markdown file.
        Args:
            output_filepath: Path to the markdown output file.
            substep_title: The title of the substep where the error occurred.
            error_message: The detailed error message.
        """
        logger.info(f"Appending error for substep '{substep_title}' to {output_filepath}. Error snippet: {error_message[:200]}")
        try:
            # Slugify substep_title for a unique anchor link, e.g. "Error in My Step" -> "error-my-step"
            # This follows a common pattern for generating URL-friendly slugs.
            s = substep_title.lower()
            # Replace spaces and common separators (like underscore or slash) with hyphens
            s = s.replace(" ", "-").replace("_", "-").replace("/", "-")
            
            # Keep only alphanumeric characters and hyphens.
            # This helps prevent issues with special characters in HTML anchors.
            raw_slug = "".join(char for char in s if char.isalnum() or char == '-')
            
            # Remove consecutive hyphens and leading/trailing hyphens that might result from replacements.
            # e.g., "---foo--bar---" becomes "foo-bar"
            slug_parts = [part for part in raw_slug.split('-') if part] # Filter out empty strings from split
            slug_substep_title = "-".join(slug_parts)
            
            # Handle cases where the title might become empty or too generic after slugification
            # (e.g., title was "!!!", " - ", or just non-alphanumeric characters).
            if not slug_substep_title:
                slug_substep_title = "generic-error-report" # Provide a descriptive fallback slug
            
            anchor = f"error-{slug_substep_title}"

            markdown_content = f"<a name=\"{anchor}\"></a>\n"  # HTML anchor
            markdown_content += f"## Error in Substep: {substep_title}\n\n"
            markdown_content += "**An error occurred during the processing of this substep, and detailed content could not be generated.**\n\n"
            markdown_content += "**Error Details:**\n"
            markdown_content += "```\n"
            markdown_content += f"{error_message.strip()}\n"
            markdown_content += "```\n\n"
            markdown_content += "---\n\n"  # Separator

            with open(output_filepath, 'a', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.debug(f"Successfully appended error notice for substep '{substep_title}' to {output_filepath}.")
        except IOError as ioe:
            logger.error(f"IOError appending error notice for substep '{substep_title}' to {output_filepath}: {ioe}", exc_info=True)
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error appending error notice for substep '{substep_title}' to {output_filepath}: {e}", exc_info=True)

    async def append_substep_analysis_to_markdown(self, output_filepath: str, step_number: int, substep_title: str, analysis_json: Dict[str, Any]) -> None:
        """
        Formats a substep's analysis JSON into markdown and appends it to the output file.
        Args:
            output_filepath: Path to the markdown output file.
            step_number: The number of the substep.
            substep_title: The title of the substep (e.g., "Define Lead Generation Objectives").
            analysis_json: The structured JSON analysis for this substep.
        """
        logger.info(f"Appending analysis for substep {step_number}: '{substep_title}' to {output_filepath}")
        try:
            # Slugify title for anchor link, consistent with TOC generation
            # Ensure anchor generation is robust for various title inputs
            s = substep_title.lower()
            # Replace spaces and common separators (like underscore or slash) with hyphens
            s = s.replace(" ", "-").replace("_", "-").replace("/", "-")
            # Keep only alphanumeric characters and hyphens.
            raw_slug = "".join(char for char in s if char.isalnum() or char == '-')
            # Remove consecutive hyphens and leading/trailing hyphens
            slug_parts = [part for part in raw_slug.split('-') if part]
            anchor = "-".join(slug_parts)
            if not anchor: # Fallback if title was all non-alphanumeric
                anchor = f"substep-{step_number}"

            markdown_content = f"<a name=\"{anchor}\"></a>\n"  # HTML anchor for TOC
            markdown_content += f"## {step_number}. {substep_title}\n\n"
            
            formatted_analysis = format_json_to_markdown(analysis_json) # Uses the global function
            if not formatted_analysis.strip():
                formatted_analysis = "No detailed analysis content was generated for this substep.\n"
                
            markdown_content += formatted_analysis
            markdown_content += "\n---\n\n"  # Separator between substeps

            # The original function uses synchronous file I/O.
            # This async method wraps sync I/O. For true async, consider 'aiofiles'.
            with open(output_filepath, 'a', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.debug(f"Successfully appended analysis for substep {step_number}: '{substep_title}'.")
        except IOError as ioe:
            logger.error(f"IOError appending substep analysis for {step_number}: '{substep_title}' to {output_filepath}: {ioe}", exc_info=True)
        except Exception as e:
            logger.error(f"Error appending substep analysis for {step_number}: '{substep_title}' to {output_filepath}: {e}", exc_info=True)

# Example Usage (Commented out as before)
# Needs to be updated to reflect new engine structure if uncommented
# For example, showing how write_document_header, then multiple calls to
# append_substep_analysis_to_markdown (externally), then write_document_conclusion would work.
#
# import asyncio
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#
#     # Mock Model Handler
#     class MockLLMHandler:
#         async def execute(self, prompt: str, session: aiohttp.ClientSession, iteration: int = 1) -> Dict[str, Any]:
#             logger.debug(f"MockLLMHandler received prompt (snippet): {prompt[:200]}...")
#             if "introduction" in prompt.lower():
#                 return {'status': 'success', 'content': 'This is a mock introduction.'}
#             elif "conclusion" in prompt.lower():
#                 return {'status': 'success', 'content': '## Conclusion\nThis is a mock conclusion.'}
#             return {'status': 'error', 'error_message': 'Unknown prompt type for mock'}
#         def is_ready(self) -> bool: return True
#
#     mock_config = {}
#     mock_llm_handler = MockLLMHandler()
#     engine = SynthesisEngine(mock_config, mock_llm_handler)
#
#     mock_output_file = "mock_architectural_plan.md"
#     mock_main_query = "Develop a new CRM System"
#     mock_substep_titles = [
#         "Define CRM Objectives",
#         "Identify Key CRM Features",
#         "Design Data Model for CRM"
#     ]
#     mock_substep_analyses = [
#         {"step_info": {"name": "Define CRM Objectives"}, "consolidated_content": {"Summary/Overview": "Objectives are X, Y, Z.", "Key Considerations": "Consider A, B, C."}},
#         {"step_info": {"name": "Identify Key CRM Features"}, "consolidated_content": {"Recommended Features": ["Contact Management", "Lead Tracking"], "Rationale": "Essential for sales."}},
#         {"step_info": {"name": "Design Data Model for CRM"}, "consolidated_content": {"Entities": ["Contacts", "Deals"], "Relationships": "1-to-N"}}
#     ]
#
#     async def run_mock_synthesis_flow():
#         # 1. Write header
#         await engine.write_document_header(mock_output_file, mock_main_query, mock_substep_titles)
#
#         # 2. Append each substep (this would happen in main.py loop)
#         for i, analysis_output in enumerate(mock_substep_analyses):
#             substep_title = analysis_output["step_info"]["name"]
#             # In real scenario, analysis_json would be the 'consolidated_analysis' from AnalyzerEngine
#             analysis_json = analysis_output["consolidated_content"]
#             append_substep_analysis_to_markdown(substep_title, analysis_json, mock_output_file)
#
#         # 3. Write conclusion
#         await engine.write_document_conclusion(mock_output_file, mock_main_query)
#
#         print(f"Mock architectural plan generated: {mock_output_file}")
#         # You can open and check mock_architectural_plan.md
#
#     # asyncio.run(run_mock_synthesis_flow()) # Keep commented for production code
#
#     # Example mock data based on the new AnalyzerEngine output structure
#     mock_consolidated_outputs = [
#         {
#             'step_info': {'step_number': 1, 'name': 'Define Requirements', 'prompt': '...'},
#             'status': 'success',
#             'consolidated_content': {
#                 'Summary/Overview': '...',
#                 'Key Considerations/Factors': '...',
#                 # ... other categories
#             },
#             'raw_analysis_response': '...'
#         },
#         {
#             'step_info': {'step_number': 2, 'name': 'Design Data Flow', 'prompt': '...'},
#             'status': 'success',
#             'consolidated_content': {
#                 'Summary/Overview': '...',
#                 'Recommended Approach/Design': '...',
#                 # ... other categories
#             },
#             'raw_analysis_response': '...'
#         },
#         # Add a step with an error for testing error handling
#         {
#             'step_info': {'step_number': 3, 'name': 'Select Tech Stack', 'prompt': '...'},
#             'status': 'error',
#             'consolidated_content': {'Summary/Overview': 'Error occurred during analysis.'},
#             'raw_analysis_response': '...',
#             'error_message': 'LLM failed to respond.'
#         }
#     ]
#
#     # Need a mock SynthesisHandler for testing
#     class MockSynthesisHandler:
#         async def execute(self, prompt: str, session: aiohttp.ClientSession, iteration: int = 1) -> Dict[str, Any]:
#             print(f"MockSynthesisHandler received prompt (snippet): {prompt[:500]}...")
#             # Simulate a successful response
#             return {'status': 'success', 'content': 'This is the final synthesized plan based on the provided analysis results.'}
#
#         def is_ready(self) -> bool:
#             return True
#
#     mock_config = {} # Placeholder config
#     mock_handler = MockSynthesisHandler()
#     synthesizer = SynthesisEngine(mock_config, mock_handler)
#
#     async def run_mock_synthesis():
#          final_plan = await synthesizer.synthesize(mock_consolidated_outputs)
#          print("\n--- Final Synthesized Plan ---")
#          print(final_plan)
#
#     asyncio.run(run_mock_synthesis())