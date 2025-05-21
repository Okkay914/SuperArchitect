import logging
import json
import yaml # Retained for other parts of the application, if any.
import re
from typing import Dict, List, Any, Optional # Tuple removed as _validate_json_structure is removed
import aiohttp

from ..models.base import ModelHandler

# Define standard architectural sections (kept for potential future use or existing code)
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

# Define standard categories for per-step analysis
STANDARD_CATEGORIES = [
    "Summary/Overview",
    "Key Considerations/Factors",
    "Recommended Approach/Design",
    "Components and Structure",
    "Technical Recommendations",
    "Implementation Steps/Actions",
    "Pros and Cons/Trade-offs",
    "Further Research/Open Questions",
]


class AnalyzerEngine:
    """
    Generates Markdown content for standard analytical categories for each substep.
    """
    def __init__(self, config: Dict[str, Any], analyzer_handler: Optional[ModelHandler], standard_categories: List[str]):
        """
        Initializes the AnalyzerEngine.

        Args:
            config: The application configuration dictionary.
            analyzer_handler: The instantiated model handler for the analyzer model.
            standard_categories: A list of standard categories for segmentation.
        """
        self.config = config
        self.analyzer_handler = analyzer_handler
        self.standard_categories = standard_categories if standard_categories is not None else STANDARD_CATEGORIES
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG) # Or from config
        self.logger.info(f"AnalyzerEngine initialized with {len(self.standard_categories)} standard categories. Analyzer handler: {analyzer_handler.__class__.__name__ if analyzer_handler else 'None'}")
        if not self.analyzer_handler or not self.analyzer_handler.is_ready():
             self.logger.warning("Analyzer handler is not provided or not ready. Markdown generation will likely fail or use defaults.")

    def _clean_llm_markdown_output(self, raw_output: str) -> str:
        """
        Cleans raw LLM output to extract Markdown content.
        Strips common preamble/postamble and markdown code blocks if they wrap the whole content.
        """
        self.logger.debug(f"Raw input for Markdown cleaning (first 200 chars): {raw_output[:200]}...")
        text_to_process = raw_output.strip()

        if not text_to_process:
            self.logger.warning("_clean_llm_markdown_output: Input is empty or whitespace.")
            return ""

        # Remove common preambles (case-insensitive start)
        preambles = [
            "Here's the Markdown content:", "Here is the Markdown content:", "Here is the Markdown:",
            "Here's the Markdown section:", "Here is the Markdown section:",
            "Certainly, here is the Markdown for that section:", "Okay, here's the Markdown:",
            "```markdown", "```" # Also strip if it starts directly with a fence
        ]
        # More aggressive preamble stripping
        original_length = len(text_to_process)
        for preamble_pattern in preambles:
            # Check if text_to_process starts with preamble_pattern (case insensitive for text part)
            if re.match(f"^{re.escape(preamble_pattern)}", text_to_process, re.IGNORECASE):
                # Find the actual length of the matched preamble (could be shorter if only ``` was matched)
                match = re.match(f"^{re.escape(preamble_pattern)}", text_to_process, re.IGNORECASE)
                if match:
                    actual_preamble_len = len(match.group(0))
                    text_to_process = text_to_process[actual_preamble_len:].lstrip()
                    self.logger.debug(f"_clean_llm_markdown_output: Removed preamble matching '{preamble_pattern}'.")
                    break # Stop after removing one preamble type

        # Remove common postambles (case-insensitive end)
        postambles = [
            "Let me know if you need anything else!",
            "Feel free to ask if you have more questions."
            # Add more common postambles
        ]
        for postamble in postambles:
            if text_to_process.lower().endswith(postamble.lower()):
                text_to_process = text_to_process[:-len(postamble)].rstrip()
                self.logger.debug(f"_clean_llm_markdown_output: Removed postamble '{postamble}'.")
                break
        
        # Handle potential wrapping ```markdown ... ``` or ``` ... ``` blocks
        # This regex looks for content optionally starting/ending with a fence.
        # If the LLM ONLY returns the Markdown, it might not be fenced.
        # If it fences its pure Markdown, we want the content.
        # If it fences Markdown + other text, this might be too simple.
        # The goal is to get the raw Markdown.
        
        # If after preamble stripping, it still looks like a fenced block, extract from it.
        fence_match = re.match(r"^\s*```(?:markdown)?\s*([\s\S]*?)\s*```\s*$", text_to_process, re.DOTALL | re.IGNORECASE)
        if fence_match:
            content_within_fences = fence_match.group(1).strip()
            self.logger.debug(f"Content extracted from within a fully wrapping triple-backtick Markdown block: '{content_within_fences[:200]}...'")
            text_to_process = content_within_fences
        else:
            # If not fully wrapped, ensure hanging backticks are removed if they were part of preamble/postamble stripping.
            text_to_process = text_to_process.strip()
            if text_to_process.startswith("```markdown"): text_to_process = text_to_process[len("```markdown"):].lstrip()
            if text_to_process.startswith("```"): text_to_process = text_to_process[len("```"):].lstrip()
            if text_to_process.endswith("```"): text_to_process = text_to_process[:-len("```")].rstrip()


        cleaned_markdown = text_to_process.strip()
        if cleaned_markdown != raw_output.strip() or original_length != len(text_to_process): # Log if changes were made
             self.logger.debug(f"_clean_llm_markdown_output: Result after cleaning (first 200 chars): '{cleaned_markdown[:200]}...'")
        return cleaned_markdown

    def _clean_llm_json_output(self, raw_output: str) -> str:
        """
        Cleans raw LLM output to extract a JSON string.
        Strips common preamble/postamble and code blocks (e.g., ```json ... ```).
        """
        self.logger.debug(f"Raw input for JSON cleaning (first 200 chars): {raw_output[:200]}...")
        text_to_process = raw_output.strip()

        if not text_to_process:
            self.logger.warning("_clean_llm_json_output: Input is empty or whitespace.")
            return ""

        # Attempt to find JSON within triple backticks
        # Regex to find ```json ... ``` or ``` ... ```
        # It captures the content within the innermost backticks if nested, or the outermost if not.
        # DOTALL allows . to match newlines.
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", text_to_process, re.DOTALL | re.IGNORECASE)
        if match:
            json_string = match.group(1).strip()
            self.logger.debug(f"_clean_llm_json_output: Extracted JSON from triple-backtick block: '{json_string[:200]}...'")
            return json_string
        
        # If not in backticks, try to find the first '{' or '[' and last '}' or ']'
        # This is a more brittle approach but can work if the LLM just returns raw JSON.
        # It assumes the JSON is the primary content.
        
        # First, remove common textual preambles that might precede raw JSON
        preambles = [
            "Here's the JSON response:", "Here is the JSON response:", "Here is the JSON:",
            "JSON Response:", "Response:", "Output:",
            "```json", "```" # In case they weren't caught by the regex above or are partial
        ]
        original_length_before_preamble_strip = len(text_to_process)
        for preamble_pattern in preambles:
            if re.match(f"^{re.escape(preamble_pattern)}", text_to_process, re.IGNORECASE):
                match_preamble = re.match(f"^{re.escape(preamble_pattern)}", text_to_process, re.IGNORECASE)
                if match_preamble:
                    actual_preamble_len = len(match_preamble.group(0))
                    text_to_process = text_to_process[actual_preamble_len:].lstrip()
                    self.logger.debug(f"_clean_llm_json_output: Removed preamble matching '{preamble_pattern}'.")
                    break
        
        # Strip trailing backticks if any remain
        if text_to_process.endswith("```"):
            text_to_process = text_to_process[:-3].rstrip()
            self.logger.debug("_clean_llm_json_output: Removed trailing triple backticks.")

        # After stripping, check if it looks like a JSON object or array
        # This is a basic check; actual JSON validation happens during parsing.
        text_to_process = text_to_process.strip() # Ensure it's stripped again
        if (text_to_process.startswith('{') and text_to_process.endswith('}')) or \
           (text_to_process.startswith('[') and text_to_process.endswith(']')):
            self.logger.debug(f"_clean_llm_json_output: Content appears to be JSON after initial cleaning: '{text_to_process[:200]}...'")
            return text_to_process
        
        self.logger.warning(f"_clean_llm_json_output: Could not confidently extract JSON. Returning potentially uncleaned/partial data or empty string if nothing plausible found. Original (first 200): {raw_output[:200]}... Processed (first 200): {text_to_process[:200]}")
        # If no clear JSON structure is found after attempts, return the processed text.
        # The calling function (json.loads) will ultimately determine if it's valid.
        # It's better to return something that *might* be parsable than nothing if unsure.
        # However, if it's clearly not JSON (e.g., just conversational text), an empty string might be better.
        # For now, let's return what we have, as the original code in main.py checks for empty string from cleaner.
        # If it's still not parsable, json.loads will fail, which is handled.
        if not text_to_process.strip().startswith('{') and not text_to_process.strip().startswith('['):
             self.logger.debug("_clean_llm_json_output: Processed text does not start with { or [. Returning empty string.")
             return "" # Return empty if it clearly doesn't look like JSON start

        return text_to_process.strip()

    def _add_retry_instructions_to_prompt(
        self,
        original_prompt_text: str,
        category_name: str,
        attempt_num: int,
        error_type: str,
        error_details: str,
        raw_content_if_any: Optional[str]
    ) -> str:
        """Helper to add retry instructions to the LLM prompt for Markdown generation."""
        retry_instruction = f"\n\n--- CRITICAL RETRY INSTRUCTION (Attempt {attempt_num} for category '{category_name}' FAILED) ---\n"
        retry_instruction += f"The previous attempt to generate Markdown for the '{category_name}' section failed due to: **{error_type}**.\n"
        retry_instruction += f"Error details: {error_details}\n"
        if raw_content_if_any:
            raw_content_str = str(raw_content_if_any)
            retry_instruction += f"The problematic raw content from the FAILED attempt was (showing snippet up to 500 chars):\n```\n{raw_content_str[:500]}...\n```\n"
        else:
            retry_instruction += "There was no raw content available from the failed attempt to show.\n"

        retry_instruction += f"\n**Please regenerate the response for the '{category_name}' section, ensuring you output ONLY well-structured Markdown content.**\n"
        retry_instruction += "Key reminders for clean Markdown output:\n"
        retry_instruction += "1.  Your ENTIRE response for this section MUST be well-structured Markdown.\n"
        retry_instruction += "2.  DO NOT include any conversational preamble (e.g., 'Here is the markdown...') or postamble (e.g., 'Let me know if...'). Just provide the Markdown itself.\n"
        retry_instruction += "3.  Use appropriate Markdown syntax for headings (e.g., `## Section Title`), lists, bold/italic text, code blocks (e.g., ```python\\n# code\\n```), etc.\n"
        retry_instruction += f"4.  Ensure the content is comprehensive and directly addresses the requirements for the '{category_name}' section as outlined in the original prompt.\n"
        retry_instruction += "5.  Do NOT wrap your Markdown output in triple backticks (e.g. ```markdown ... ```) unless the Markdown itself is a single code block you intend to render as such. Just provide the raw Markdown flow.\n"
        retry_instruction += "--- End Critical Retry Instruction ---\n"

        return original_prompt_text + retry_instruction

    async def analyze_and_synthesize_step(
        self,
        substep_title: str,
        key_question_answers: List[str], # Kept for context, though not in example prompt directly
        overall_query_context: str,
        previous_substep_instructions: Optional[str] # This is now Markdown
    ) -> Dict[str, str]: # RETURN TYPE CHANGED to Dict[str, str]
        self.logger.info(f"Starting Markdown generation for all categories in substep: {substep_title}")

        markdown_outputs: Dict[str, str] = {}
        
        def _get_default_error_markdown_for_all_categories(error_message: str) -> Dict[str, str]:
            self.logger.error(f"analyze_and_synthesize_step returning error for all categories: {error_message} for substep '{substep_title}'")
            error_outputs: Dict[str, str] = {}
            for category_name_err in self.standard_categories:
                error_outputs[category_name_err] = f"*Content for the '{category_name_err}' section could not be generated for substep '{substep_title}': {error_message}.*"
            return error_outputs

        if not self.analyzer_handler or not self.analyzer_handler.is_ready():
            return _get_default_error_markdown_for_all_categories("Analyzer handler is not available or ready")

        if not overall_query_context:
             return _get_default_error_markdown_for_all_categories("Missing overall_query_context")
        
        # Note: key_question_answers are part of overall_query_context generation upstream,
        # or could be explicitly added to the prompt here if needed for each category.
        # For now, adhering to the simpler prompt from task description.

        max_retries_per_category = self.config.get('analyzer_llm_retries', 1) # 0 means 1 attempt, 1 means 1 retry (2 attempts total)
        num_attempts_per_category = max_retries_per_category + 1


        for category_name in self.standard_categories:
            self.logger.info(f"Generating Markdown for category: '{category_name}' of substep: '{substep_title}'")

            # Construct the base prompt for this specific category
            # previous_substep_instructions is Markdown from the previous step.
            previous_instructions_context = "N/A. This is the first substep or no previous instructions were successfully generated."
            if previous_substep_instructions and previous_substep_instructions.strip():
                 previous_instructions_context = previous_substep_instructions

            base_prompt_for_category = f"""You are an expert technical writer. For the substep titled '{substep_title}', provide a comprehensive section covering '{category_name}'.
Format your response as well-structured Markdown. Include subheadings (e.g. `### Sub-heading`), bullet points (`* item`), code examples (e.g. ```python\\n# code\\n``` if relevant for this category), etc., as appropriate.
Overall User Query Context: {overall_query_context}
Context from Previous Substep's Instructions (if any):
{previous_instructions_context}

Generate only the Markdown content for the '{category_name}' section.
Do NOT include any preamble like "Here is the markdown...", "Certainly, ...", or any explanations outside of the Markdown itself.
Do NOT wrap your entire response in triple backticks (e.g. ```markdown ... ```) unless the specific content for this category is inherently a single large code block.
"""
            # Adding key_question_answers to the prompt if available and relevant
            # This could be a global decision or per-category
            # For now, we will add it to the overall context if it is not already there by assumption
            # if key_question_answers:
            #     answers_context = "\nKey Information/Answers Relevant to this Substep:\n"
            #     for i, answer in enumerate(key_question_answers):
            #         answers_context += f"- Info {i+1}: {answer}\n"
            #     base_prompt_for_category += answers_context


            current_prompt_for_category = base_prompt_for_category
            category_markdown_generated = False
            last_error_message_for_category = "LLM call not yet attempted for category."
            raw_llm_response_for_category = None # For retry/debug info

            for attempt in range(num_attempts_per_category):
                self.logger.debug(f"LLM call for category '{category_name}', substep '{substep_title}' (Attempt {attempt + 1}/{num_attempts_per_category}) "
                                 f"Prompt length: {len(current_prompt_for_category)}, first 300 chars: {current_prompt_for_category[:300]}...")
                try:
                    # Assuming aiohttp.ClientSession is still the way to go if handler uses it
                    # The handler.execute method needs to be compatible.
                    async with aiohttp.ClientSession() as session:
                        llm_result = await self.analyzer_handler.execute(current_prompt_for_category, session)

                    if llm_result.get('status') == 'success':
                        raw_llm_response_for_category = llm_result.get('content', '')
                        self.logger.debug(f"Raw LLM response for category '{category_name}' (Attempt {attempt+1}, first 200 chars): {raw_llm_response_for_category[:200]}...")
                        
                        cleaned_markdown = self._clean_llm_markdown_output(raw_llm_response_for_category)

                        if cleaned_markdown.strip():
                            markdown_outputs[category_name] = cleaned_markdown
                            category_markdown_generated = True
                            self.logger.info(f"Successfully generated and cleaned Markdown for category '{category_name}', substep '{substep_title}' (Attempt {attempt+1}). Length: {len(cleaned_markdown)}")
                            break # Success for this category, exit attempt loop
                        else:
                            last_error_message_for_category = "LLM returned empty content after cleaning."
                            self.logger.warning(f"{last_error_message_for_category} Category '{category_name}' (Attempt {attempt + 1}). Raw response (first 200): {raw_llm_response_for_category[:200]}")
                            # This will lead to a retry if attempts remain

                    else: # LLM handler call was not 'success'
                        llm_error = llm_result.get('error_message', 'Unknown error from analyzer handler.')
                        last_error_message_for_category = f"LLM handler error: {llm_error}"
                        raw_llm_response_for_category = llm_result.get('content', None) # For retry instructions
                        self.logger.error(f"LLM execution failed for category '{category_name}' (Attempt {attempt + 1}): {last_error_message_for_category}")
                    
                    # If here, it's an LLM error or empty content; prepare for retry if attempts remain.
                    if attempt < num_attempts_per_category - 1:
                        current_prompt_for_category = self._add_retry_instructions_to_prompt(
                            original_prompt_text=base_prompt_for_category,
                            category_name=category_name,
                            attempt_num=attempt + 1, # This was the attempt that failed
                            error_type="LLM Error or Empty/Invalid Content",
                            error_details=last_error_message_for_category,
                            raw_content_if_any=raw_llm_response_for_category
                        )
                        self.logger.info(f"Preparing retry for category '{category_name}' (Attempt {attempt + 2}/{num_attempts_per_category}).")
                        # continue to next attempt in loop is implicit
                    else:
                        self.logger.error(f"Max attempts ({num_attempts_per_category}) reached for category '{category_name}', substep '{substep_title}'. Last error: {last_error_message_for_category}")
                        # Fall through to use default after loop

                except Exception as e_cat:
                    last_error_message_for_category = f"Unexpected error during LLM call: {type(e_cat).__name__} - {str(e_cat)}"
                    self.logger.error(f"{last_error_message_for_category} for category '{category_name}', substep '{substep_title}' (Attempt {attempt + 1})", exc_info=True)
                    raw_llm_response_for_category = None # Response might not exist if error was early

                    if attempt < num_attempts_per_category - 1:
                        current_prompt_for_category = self._add_retry_instructions_to_prompt(
                            original_prompt_text=base_prompt_for_category,
                            category_name=category_name,
                            attempt_num=attempt + 1,
                            error_type=f"Processing Exception: {type(e_cat).__name__}",
                            error_details=str(e_cat),
                            raw_content_if_any=raw_llm_response_for_category
                        )
                        self.logger.info(f"Preparing retry for category '{category_name}' due to exception (Attempt {attempt + 2}/{num_attempts_per_category}).")
                        # continue to next attempt in loop is implicit
                    else:
                        self.logger.error(f"Max attempts ({num_attempts_per_category}) reached after exception for category '{category_name}', substep '{substep_title}'. Last error: {last_error_message_for_category}")
                        # Fall through to use default after loop
            
            # After all attempts for a category, if not generated, use default.
            if not category_markdown_generated:
                default_md = f"*Content for the '{category_name}' section could not be generated for substep '{substep_title}' due to an error. Last reported issue: {last_error_message_for_category}.*"
                markdown_outputs[category_name] = default_md
                self.logger.warning(f"Using default Markdown for category '{category_name}', substep '{substep_title}'.")

        self.logger.info(f"Finished Markdown generation for all categories in substep: {substep_title}. Generated {len(markdown_outputs)}/{len(self.standard_categories)} successfully.")
        return markdown_outputs

    # Keep the existing analyze method for now, in case it's used elsewhere.
    # It operates on STANDARD_SECTIONS, not STANDARD_CATEGORIES.
    async def _analyze_with_llm(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Uses the configured analyzer LLM to analyze the consultation responses
        based on STANDARD_SECTIONS. Creates its own session for the API call.
        """
        self.logger.warning("Using deprecated _analyze_with_llm method based on STANDARD_SECTIONS.")
        if not self.analyzer_handler or not self.analyzer_handler.is_ready():
            self.logger.error("Analyzer handler is not available or ready.")
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
        prompt += "Ensure all double quotes within string values in the JSON are properly escaped with a backslash (\\\\\").\n\n"
        prompt += "Provide ONLY the JSON object in your response, without any introductory text or explanations outside the JSON structure. Ensure the JSON is strictly valid and well-formed, with correct punctuation (commas, colons, brackets, braces).\n"
        prompt += "JSON Response:\n"

        self.logger.debug(f"Sending analysis prompt to {self.analyzer_handler.__class__.__name__}...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    analysis_result = await self.analyzer_handler.execute(prompt, session)

                if analysis_result.get('status') == 'success':
                    content = analysis_result.get('content', '{}')
                    self.logger.debug(f"Raw LLM response content (Attempt {attempt + 1}): {content}")
                    try:
                        cleaned_content = content.strip()
                        if cleaned_content.startswith("```json"):
                            cleaned_content = cleaned_content[7:-3].strip()
                        elif cleaned_content.startswith("```"):
                             cleaned_content = cleaned_content[3:-3].strip()

                        parsed_json = json.loads(cleaned_content)

                        if isinstance(parsed_json, dict) and \
                           'summary' in parsed_json and \
                           'reasoning' in parsed_json and \
                           'segmented_architecture' in parsed_json and \
                           isinstance(parsed_json['segmented_architecture'], dict):
                            for section in STANDARD_SECTIONS:
                                parsed_json['segmented_architecture'].setdefault(section, [])
                            parsed_json['status'] = 'success'
                            return parsed_json
                        else:
                            self.logger.error(f"Analyzer LLM response is not valid JSON or missing required keys (Attempt {attempt + 1}). Content: {content[:500]}...")
                            raise ValueError("Invalid JSON structure or missing keys in analyzer response.")

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to decode JSON response from analyzer LLM (Attempt {attempt + 1}): {e}. Content: {content[:500]}...")
                        if attempt < max_retries - 1:
                            self.logger.warning(f"JSON decoding failed. Retrying with modified prompt (Attempt {attempt + 2})...")
                            prompt += f"\n\nPrevious attempt failed to decode JSON: {e}. Please provide a valid JSON object."
                            continue
                        else:
                            raise ValueError(f"Failed to decode JSON response from analyzer LLM after {max_retries} attempts: {e}")
                    except ValueError as e:
                         self.logger.error(f"Invalid structure in JSON response from analyzer LLM (Attempt {attempt + 1}): {e}. Content: {content[:500]}...")
                         raise e

                else:
                    error_msg = analysis_result.get('error_message', 'Unknown error from analyzer handler.')
                    self.logger.error(f"Analyzer LLM execution failed (Attempt {attempt + 1}): {error_msg}")
                    raise RuntimeError(f"Analyzer LLM execution failed (Attempt {attempt + 1}): {error_msg}")

            except Exception as e:
                self.logger.error(f"Error during analyzer LLM call (Attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    self.logger.warning(f"LLM call failed. Retrying (Attempt {attempt + 2})...")
                    continue
                else:
                    return {
                        'status': 'error',
                        'summary': f"Error: Failed to get analysis from LLM after {max_retries} attempts - {e}",
                        'reasoning': "Analysis could not be performed due to repeated errors.",
                        'segmented_architecture': {section: [] for section in STANDARD_SECTIONS},
                        'error_message': str(e)
                    }

        error_msg = f"Failed to get analysis from LLM after {max_retries} attempts."
        self.logger.error(error_msg)
        return {
            'status': 'error',
            'summary': error_msg,
            'reasoning': "Analysis could not be performed due to repeated errors.",
            'segmented_architecture': {section: [] for section in STANDARD_SECTIONS},
            'error_message': error_msg
        }


    async def analyze(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Main analysis method. Uses an analyzer LLM to perform analysis based on STANDARD_SECTIONS.
        No longer accepts session argument.
        """
        self.logger.warning("Using deprecated analyze method based on STANDARD_SECTIONS. Consider using analyze_and_synthesize_step instead.")
        return await self._analyze_with_llm(responses)


# Keep the example usage block, but it won't work correctly without
# providing a config and a mock handler during instantiation.
# It's better to test this via the main script execution.
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.DEBUG)
#     # ... (Example usage would need modification) ...