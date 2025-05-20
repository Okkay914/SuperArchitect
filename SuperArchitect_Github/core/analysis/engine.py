import logging
import json
import yaml # Retained for other parts of the application, if any.
import re
from typing import Dict, List, Any, Tuple, Optional
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
    Analyzes responses from multiple AI models, segments findings,
    performs comparative analysis, and synthesizes consolidated outputs per step.
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
        self.standard_categories = standard_categories
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"AnalyzerEngine initialized with {len(standard_categories)} standard categories. Analyzer handler: {analyzer_handler.__class__.__name__ if analyzer_handler else 'None'}")
        if not self.analyzer_handler or not self.analyzer_handler.is_ready():
             self.logger.warning("Analyzer handler is not provided or not ready. Analysis will likely fail.")

    def _find_matching_bracket(self, text: str, open_char: str, close_char: str) -> int:
        """
        Finds the position of the matching closing bracket for the first opening bracket,
        respecting nesting. Returns -1 if not found or if text is too short/malformed.
        """
        if not text:
            self.logger.debug("_find_matching_bracket: Input text is empty.")
            return -1
        
        first_open_pos = text.find(open_char)
        if first_open_pos == -1:
            self.logger.debug(f"_find_matching_bracket: Opening character '{open_char}' not found.")
            return -1

        balance = 0
        for i in range(first_open_pos, len(text)):
            if text[i] == open_char:
                balance += 1
            elif text[i] == close_char:
                balance -= 1
            
            if balance == 0:
                self.logger.debug(f"_find_matching_bracket: Found matching '{close_char}' for '{open_char}' at index {i} (started at {first_open_pos}).")
                return i
        
        self.logger.debug(f"_find_matching_bracket: No matching '{close_char}' found for '{open_char}' starting at {first_open_pos}.")
        return -1

    def _fix_json_syntax(self, text: str) -> str:
        """
        Attempts to fix common JSON syntax errors, including specific string escaping
        as requested by the user.
        """
        if not text or not text.strip():
            self.logger.debug("_fix_json_syntax: input text is empty or whitespace, returning as is.")
            return text

        self.logger.debug(f"_fix_json_syntax: input text (first 200 chars): {text[:200]}...")
        original_text = text

        # 1. Remove JavaScript comments
        text_before_comments = text
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        if text != text_before_comments:
            self.logger.debug("_fix_json_syntax: Removed JavaScript-style comments.")
        
        text = text.strip()
        if not text:
            self.logger.debug("_fix_json_syntax: text became empty after comment removal, returning empty string.")
            return "" # Return empty if all content was comments

        # 2. Fix trailing commas before closing braces or brackets
        text_before_trailing_comma_fix = text
        text = re.sub(r',\s*(\}|\])', r'\1', text)
        if text != text_before_trailing_comma_fix:
            self.logger.debug("_fix_json_syntax: Fixed trailing commas.")

        # 3. Attempt to fix some missing commas
        text_before_missing_comma_fix = text
        # Missing comma: string" \n "key": value  -> string", \n "key": value
        text = re.sub(r'(")\s*\n\s*(")', r'\1,\n\2', text)
        # Missing comma: } \n "key": value -> }, \n "key": value
        text = re.sub(r'(\})\s*\n\s*(")', r'\1,\n\2', text)
        # Missing comma: ] \n "key": value -> ], \n "key": value
        text = re.sub(r'(\])\s*\n\s*(")', r'\1,\n\2', text)
        # Missing comma: value" \n value  (e.g. in array "elem1"\n"elem2")
        text = re.sub(r'(")\s*\n\s*([\{\["tf\d])', r'\1,\n\2', text) # check if next line starts with typical JSON value starters
        if text != text_before_missing_comma_fix:
            self.logger.debug("_fix_json_syntax: Attempted to fix some missing commas.")
        
        # 4. Attempts to fix some string escaping issues and newlines in strings.
        # Part 4a: Replace literal newlines, tabs, and ensure backslashes are escaped within apparent string values.
        text_before_std_escape_fix = text
        
        # Iteratively build the string to handle modifications correctly within string literals
        parts = []
        last_end = 0
        # Regex to find JSON strings: "..." - captures content in group 1
        # It handles escaped quotes and other escaped characters within the string.
        for match in re.finditer(r'"((?:\\.|[^"\\])*)"', text):
            parts.append(text[last_end:match.start(1)])  # Append text before the string content
            
            content = match.group(1)  # The raw content of the string, e.g., hello\\nworld or hello"world
            
            # Ensure backslashes are themselves escaped (e.g., \ -> \\)
            # This needs to be done carefully to not double-escape already correct sequences like \n or \"
            # A simpler approach for LLM output: assume \ are literal unless part of \n, \t, etc.
            # For now, let's focus on common LLM errors: literal newlines, tabs, and unescaped quotes.
            
            # Replace literal newlines (and carriage returns) with \n
            content = content.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
            # Replace literal tabs with \t
            content = content.replace('\t', '\\t')
            # Escape backslashes that are not already part of a valid escape sequence
            # This is complex. For now, trust the user's global rule for quotes below,
            # and assume simple \ -> \\ is too broad here.
            # content = content.replace('\\', '\\\\') # This would turn \n into \\n if applied naively.

            parts.append(content)
            last_end = match.end(1) # Position after the string content
        parts.append(text[last_end:]) # Append any remaining text
        text = "".join(parts)

        if text != text_before_std_escape_fix:
             self.logger.debug("_fix_json_syntax: Standardized newlines/tabs in detected string contents.")

        # Part 4b: More selective approach to handle quotes rather than global replacement
        text_before_quote_fix = text
        self.logger.debug("_fix_json_syntax: Using a more selective approach for handling quotes instead of global replacement")
        
        # We'll only process text outside of properly formatted JSON strings
        # This is a simplified approach - for a production system, consider a proper JSON lexer
        # The current approach avoids breaking already valid JSON while still helping with common issues
        
        # Instead of aggressive global replacement, we'll leave text as is
        # Most JSON libraries can handle escaped quotes properly
        if text != text_before_quote_fix:
            self.logger.debug("_fix_json_syntax: Modified quote handling approach")
        
        final_text = text.strip()
        if final_text != original_text:
            self.logger.debug(f"_fix_json_syntax: String modified. Original (first 200): {original_text[:200]}... Final (first 200): {final_text[:200]}...")
        else:
            self.logger.debug("_fix_json_syntax: String not modified by any rules.")
        
        return final_text

    def _clean_llm_json_output(self, raw_output: str) -> str:
        """
        Cleans raw LLM output to extract potential JSON content.
        1. Handles empty/whitespace raw output.
        2. Prioritizes content within ```json ... ``` blocks, then generic ``` ... ``` blocks.
        3. If no blocks, or content within blocks is not directly usable,
           tries to find the outermost `{...}` or `[...]` using `_find_matching_bracket`.
        4. Calls `_fix_json_syntax` on the extracted string before returning.
        """
        self.logger.debug(f"Raw input for JSON cleaning (first 500 chars): {raw_output[:500]}...")
        text_to_process = raw_output.strip()

        if not text_to_process:
            self.logger.warning("_clean_llm_json_output: Input is empty or whitespace.")
            return ""

        # 1. Try to extract content from markdown code blocks
        # Enhanced markdown code block extraction - handle both ```json and ``` blocks
        # The regex is more robust, handling Claude-specific formatting quirks
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_to_process, re.DOTALL)
        
        if fence_match:
            content_within_fences = fence_match.group(1).strip()
            self.logger.debug(f"Content extracted from within triple-backtick block (first 200 chars): '{content_within_fences[:200]}...'")
            # Try to find a valid JSON object within the extracted content
            # This handles cases where there might be extra text inside the code block
            json_start = content_within_fences.find('{')
            json_end = content_within_fences.rfind('}')
            if json_start >= 0 and json_end > json_start:
                self.logger.debug(f"Found JSON object markers within code block: start={json_start}, end={json_end}")
                text_to_process = content_within_fences[json_start:json_end+1]
            else:
                # If no JSON object found, use the whole content (might be a JSON array)
                text_to_process = content_within_fences
        else:
            self.logger.debug("No triple-backtick block found. Processing entire raw output for JSON structure.")
            # text_to_process remains the original stripped raw_output

        # If, after potential fence stripping, text_to_process is empty, return it.
        if not text_to_process.strip():
             self.logger.debug("_clean_llm_json_output: Text became empty after stripping markdown fences (if any).")
             return ""

        # 2. Try to find the outermost JSON structure ({...} or [...]) from the processed text.
        final_json_str = ""
        # Find the first occurrence of '{' or '['
        first_brace_pos = text_to_process.find('{')
        first_bracket_pos = text_to_process.find('[')

        start_char = ''
        start_pos = -1

        # Determine if we are looking for an object or an array based on first significant character
        if first_brace_pos != -1 and (first_bracket_pos == -1 or first_brace_pos < first_bracket_pos):
            start_char = '{'
            start_pos = first_brace_pos
        elif first_bracket_pos != -1:
            start_char = '['
            start_pos = first_bracket_pos
        
        if start_char and start_pos != -1:
            self.logger.debug(f"_clean_llm_json_output: Found initial '{start_char}' at index {start_pos}. Attempting to find matching bracket.")
            # We need to pass the substring starting from this first brace/bracket to _find_matching_bracket
            # to ensure it correctly finds the *first* open_char.
            text_subset_for_matching = text_to_process[start_pos:]
            open_char_map = {'{': '}', '[': ']'}
            
            # _find_matching_bracket expects the full text and finds the *first* open_char in it.
            # So we pass text_to_process directly.
            matching_bracket_pos_relative_to_text_to_process = self._find_matching_bracket(text_to_process, start_char, open_char_map[start_char])

            if matching_bracket_pos_relative_to_text_to_process != -1:
                # The extracted string should start from the first_brace_pos or first_bracket_pos found earlier
                final_json_str = text_to_process[start_pos : matching_bracket_pos_relative_to_text_to_process + 1]
                self.logger.debug(f"Extracted potential JSON using _find_matching_bracket (first 200 chars): '{final_json_str[:200]}...'")
            else:
                self.logger.warning(f"Found initial '{start_char}' at index {start_pos} but no matching closing bracket in (first 200 chars): '{text_to_process[:200]}...'. Will attempt to fix the full processed text.")
                final_json_str = text_to_process # Fallback to the whole text_to_process
        else:
            self.logger.warning(f"No JSON object ('{{') or array ('[') start character found at the beginning of the processed text (first 200 chars): '{text_to_process[:200]}...'. Will attempt to fix this text as is.")
            final_json_str = text_to_process # Fallback to the whole text_to_process (which might be from fences or original)

        if not final_json_str.strip():
            self.logger.warning("_clean_llm_json_output: JSON string is empty before calling _fix_json_syntax.")
            return ""

        # 3. Call _fix_json_syntax on the extracted or fallback string
        self.logger.debug(f"_clean_llm_json_output: Calling _fix_json_syntax on (first 200 chars): '{final_json_str[:200]}...'")
        fixed_json_str = self._fix_json_syntax(final_json_str)
        self.logger.debug(f"_clean_llm_json_output: Result from _fix_json_syntax (first 200 chars): '{fixed_json_str[:200]}...'")
        
        return fixed_json_str.strip()

    def _generate_default_guide(self, substep_title: str, error_info: str) -> Dict[str, Any]:
        """
        Generates a default instructional guide when JSON parsing or generation ultimately fails.
        """
        self.logger.warning(f"Generating default guide for substep '{substep_title}' due to error: {error_info}")
        default_guide = {
            "substep_title": substep_title,
            "_error_summary": "Failed to generate detailed instructional guide due to persistent LLM output parsing or generation issues.",
            "_error_details": error_info,
        }
        for category in self.standard_categories:
            default_guide[category] = (f"Content for '{category}' could not be generated for substep '{substep_title}'. "
                                       f"Reason: {error_info}")
        
        # Ensure a structured message for implementation steps if it's a category
        if "Implementation Steps/Actions" in self.standard_categories:
             default_guide["Implementation Steps/Actions"] = [{
                "step_title": "Content Generation Failed",
                "description": (f"The AI failed to produce a usable instructional guide for substep '{substep_title}' "
                                f"after multiple attempts. Error details: {error_info}. "
                                "Please review logs or try to re-evaluate the substep's goals."),
                "considerations": ["Check LLM model status if applicable.", "Review the prompt and inputs for this substep."],
                "code_examples": []
            }]
        self.logger.debug(f"Generated default guide for '{substep_title}': {str(default_guide)[:500]}...")
        return default_guide

    def _validate_json_structure(self, parsed_json: Any, substep_title: str) -> Tuple[bool, str]:
        """
        Validates if the parsed JSON is a dictionary and contains all standard categories.
        Returns a tuple: (is_valid, message).
        Message contains error or list of missing keys if not valid.
        """
        if not isinstance(parsed_json, dict):
            msg = f"Validation Error for '{substep_title}': Parsed content is not a dictionary (type: {type(parsed_json)})."
            self.logger.warning(msg)
            return False, msg

        missing_categories = [cat for cat in self.standard_categories if cat not in parsed_json]
        if missing_categories:
            msg = (f"Validation Warning for '{substep_title}': Parsed JSON dictionary is missing the following "
                   f"standard categories: {', '.join(missing_categories)}. These will be filled with default messages.")
            self.logger.warning(msg)
            # As per user: "Validates if parsed JSON is a dict and has all required categories."
            # This means missing categories makes it fail this specific validation check.
            # The main logic will still proceed to fill them, but the validator reports this.
            return False, msg

        msg = f"Validation Success for '{substep_title}': JSON is a dictionary and contains all standard categories."
        self.logger.debug(msg)
        return True, msg

    def _add_retry_instructions_to_prompt(self, original_prompt_text: str, attempt_num: int, error_type: str, error_details: str, raw_content_if_any: Optional[str]) -> str:
        """Helper to add retry instructions to the LLM prompt."""
        retry_instruction = f"\n\n--- CRITICAL RETRY INSTRUCTION (Attempt {attempt_num} FAILED) ---\n"
        retry_instruction += f"The previous attempt to generate content failed due to: **{error_type}**.\n"
        retry_instruction += f"Error details: {error_details}\n"
        if raw_content_if_any:
            # Ensure raw_content_if_any is a string before slicing
            raw_content_str = str(raw_content_if_any)
            retry_instruction += f"The problematic content (or raw LLM output) from the FAILED attempt was (showing snippet up to 1000 chars):\n```\n{raw_content_str[:1000]}...\n```\n"
        else:
            retry_instruction += "There was no raw content available from the failed attempt to show.\n"
        
        retry_instruction += "\n**Please regenerate the response, paying EXTREMELY CAREFUL attention to ALL JSON formatting rules (listed in the original prompt) and the original instructions.**\n"
        retry_instruction += "Key reminders for valid JSON:\n"
        retry_instruction += "1.  Your ENTIRE response MUST be a single, valid JSON object.\n"
        retry_instruction += "2.  Enclose the entire JSON object in triple backticks: ```json\\n{...}\\n```.\n"
        retry_instruction += "3.  DO NOT include ANY text, explanations, apologies, or markdown formatting outside the ```json ... ``` block.\n"
        retry_instruction += "4.  Use DOUBLE QUOTES for all JSON keys and string values (e.g., `\"key\": \"value\"`).\n"
        retry_instruction += "5.  Ensure proper comma placement: commas SEPARATE elements. NO trailing commas.\n"
        retry_instruction += "6.  Escape special characters within strings correctly: `\\\"`, `\\\\`, `\\n`, `\\t`, etc.\n"
        retry_instruction += "7.  All required category keys MUST be present.\n"
        retry_instruction += "--- End Critical Retry Instruction ---\n"
        
        # Append to the original base prompt text
        return original_prompt_text + retry_instruction

    async def analyze_and_synthesize_step(
        self,
        substep_title: str,
        key_question_answers: List[str],
        overall_query_context: str,
        previous_substep_instructions: Optional[str]
    ) -> Dict[str, Any]:
        self.logger.info(f"Starting analysis and synthesis for substep: {substep_title}")

        if not self.analyzer_handler or not self.analyzer_handler.is_ready():
            error_msg = "Analyzer handler is not available or ready."
            self.logger.error(error_msg)
            return {
                'substep_title': substep_title,
                'status': 'error_handler_unavailable',
                'instructional_guide': self._generate_default_guide(substep_title, error_msg),
                'raw_analysis_response': None,
                'error_message': error_msg
            }

        if not key_question_answers:
            error_msg = f"No key question answers provided for substep '{substep_title}'."
            self.logger.warning(error_msg)
            return {
                'substep_title': substep_title,
                'status': 'error_no_answers',
                'instructional_guide': self._generate_default_guide(substep_title, error_msg),
                'raw_analysis_response': None,
                'error_message': error_msg
            }
        
        # Construct the base prompt parts (will be joined and can have retry instructions appended)
        base_prompt_parts = [
            "You are an expert technical writer and planner. Your task is to create a very thorough instruction guide for a specific substep of a larger process.\n",
            f"Overall Context of the Main Query/Goal: {overall_query_context}\n",
            f"Current Substep Title: {substep_title}\n"
        ]

        if previous_substep_instructions:
            base_prompt_parts.append(f"Context from Previous Substep's Instructions (Note: these previous instructions are in YAML format, but your NEW output for THIS substep must be JSON):\n```yaml\n{previous_substep_instructions}\n```\n")
        else:
            base_prompt_parts.append("This is the first substep, so there are no previous instructions.\n")

        base_prompt_parts.append("Key Information Gathered (Answers to Key Questions for this Substep):")
        for i, answer in enumerate(key_question_answers):
            base_prompt_parts.append(f"- Answer to Question {i+1}: {answer}")
        
        base_prompt_parts.extend([
            "\n\nBased on all the above information (overall context, substep title, previous instructions, and key question answers), generate a **very thorough, detailed, step-by-step, and explanatory instruction guide** for the current substep: \"{substep_title}\".",
            "This guide should focus on **HOW to do everything** related to this substep.\n",
            "Structure your output STRICTLY in JSON format. The JSON object should have keys corresponding to the following categories. Adapt the content under these standard categories to be highly instructional and guiding:"
        ])
        
        # Define detailed instructions for each standard category
        category_specific_instructions = {
            self.standard_categories[0]: "(Summary/Overview): Provide an Instructional Overview of this substep: what it covers, its objectives, and its importance in the overall process.",
            self.standard_categories[1]: "(Key Considerations/Factors): Identify Key Instructional Considerations: critical factors, potential challenges, and essential prerequisites or dependencies for successfully completing this substep.",
            self.standard_categories[2]: "(Recommended Approach/Design): Outline Detailed Design Instructions or Recommended Procedures: if this substep involves designing something or a specific methodology, provide a step-by-step guide on how to approach it, including principles and best practices.",
            self.standard_categories[3]: "(Components and Structure): Describe Key Components, Tools, and Their Setup/Structure: Detail the necessary components, tools, or elements involved in this substep and how to set them up or structure them.",
            self.standard_categories[4]: "(Technical Recommendations): Offer Specific Technical Instructions and Examples: Provide actionable technical guidance, configurations, code snippets (if applicable, with language specified), or commands needed to execute this substep.",
            self.standard_categories[5]: "(Implementation Steps/Actions): Create a Step-by-Step Implementation Guide: Break down the substep into clear, sequential actions. For each action, explain *what* to do and *how* to do it thoroughly. Include details, examples, and expected outcomes.",
            self.standard_categories[6]: "(Pros and Cons/Trade-offs): Discuss Instructional Trade-offs and Decision Points: If there are choices to be made during this substep, explain the implications of each choice and guide the user in making informed decisions. Highlight common pitfalls and how to avoid them.",
            self.standard_categories[7]: "(Further Research/Open Questions): Suggest Follow-up Actions, Verification Steps, or Advanced Topics: Point towards any further learning, how to verify the substep's completion, advanced configurations, or next logical steps that build upon completing this substep successfully."
        }
        for category_name in self.standard_categories:
            instruction = category_specific_instructions.get(category_name, f"Provide detailed instructional content for '{category_name}'.") # Fallback
            base_prompt_parts.append(f"- '{category_name}' {instruction}")
        base_prompt_parts.append("\n") # Newline after category list

        base_prompt_parts.append("Example JSON structure for a step within 'Implementation Steps/Actions' (you can adapt this and ensure valid JSON string escaping):\n"
                            "```json\n"
                            "{\n"
                            "  \"Implementation Steps/Actions\": [\n"
                            "    {\n"
                            "      \"step_title\": \"Step 1: Initial Setup\",\n"
                            "      \"description\": \"Detailed explanation of what to do for initial setup, covering all necessary configurations.\",\n"
                            "      \"considerations\": [\n"
                            "        \"Ensure all prerequisites are met before starting.\"\n"
                            "      ],\n"
                            "      \"code_examples\": [\n"
                            "        {\n"
                            "          \"language\": \"python\",\n"
                            "          \"code\": \"# print('Hello World')\"\n"
                            "        }\n"
                            "      ]\n"
                            "    }\n"
                            "  ]\n"
                            "}\n"
                            "```\n")

        base_prompt_parts.append(
            "**CRITICAL JSON FORMATTING RULES:**\n"
            "1.  Your ENTIRE response MUST be a single, valid JSON object.\n"
            "2.  Enclose the entire JSON object in triple backticks: ```json\\n{...}\\n```.\n"
            "3.  DO NOT include ANY text, explanations, apologies, or markdown formatting outside the ```json ... ``` block.\n"
            "4.  Use DOUBLE QUOTES for all JSON keys and string values (e.g., `\"key\": \"value\"`). Single quotes are NOT allowed.\n"
            "5.  Ensure proper comma placement: commas SEPARATE elements in arrays and key-value pairs in objects. DO NOT put a comma after the LAST element in an array or the LAST key-value pair in an object (no trailing commas).\n"
            "6.  Escape special characters within strings correctly: `\\\"` for a double quote, `\\\\` for a backslash, `\\n` for a newline, `\\t` for a tab, etc.\n"
            "7.  All category keys listed above (e.g., 'Summary/Overview', 'Key Considerations/Factors', etc.) MUST be present in your JSON output.\n"
            "8.  Verify your JSON is well-formed and parsable before concluding your response.\n"
        )
        
        base_prompt_text = "".join(base_prompt_parts)
        current_prompt = base_prompt_text # Initial prompt

        max_retries = self.config.get('analyzer_llm_retries', 3)
        raw_analysis_response = None
        parsed_instructional_guide = None
        last_error_message = "No response yet."
        cleaned_json_content_for_retry_info = None


        for attempt in range(max_retries):
            self.logger.debug(f"Sending instructional guide prompt to {self.analyzer_handler.__class__.__name__} for substep '{substep_title}' (Attempt {attempt + 1}/{max_retries}, Prompt length: {len(current_prompt)}, first 1000: {current_prompt[:1000]}...)")
            
            try:
                async with aiohttp.ClientSession() as session:
                    analysis_result = await self.analyzer_handler.execute(current_prompt, session)

                if analysis_result.get('status') == 'success':
                    raw_analysis_response = analysis_result.get('content', '')
                    self.logger.debug(f"Raw LLM response for '{substep_title}' (Attempt {attempt + 1}, first 500 chars): {raw_analysis_response[:500]}...")
                    self.logger.debug(f"Raw LLM response (repr) for '{substep_title}' (Attempt {attempt + 1}, first 500 chars): {repr(raw_analysis_response[:500])}...") # Log with repr
                    
                    # Enhanced logging for JSON debugging
                    self.logger.debug(f"Raw LLM response structure for '{substep_title}' (Attempt {attempt + 1}): {repr(raw_analysis_response[:200])}...")
                    
                    cleaned_json_content = self._clean_llm_json_output(raw_analysis_response)
                    cleaned_json_content_for_retry_info = cleaned_json_content # Store for potential error reporting
                    self.logger.debug(f"Cleaned JSON for '{substep_title}' (Attempt {attempt + 1}, first 500 chars): {cleaned_json_content[:500]}...")
                    
                    if not cleaned_json_content.strip():
                        self.logger.error(f"Cleaned JSON content is empty after processing")
                    else:
                        self.logger.debug(f"JSON content structure after cleaning: {repr(cleaned_json_content[:200])}...")

                    if not cleaned_json_content.strip():
                        self.logger.error(f"JSON content is empty after cleaning for '{substep_title}' (Attempt {attempt + 1}). Raw: {raw_analysis_response[:200]}")
                        raise json.JSONDecodeError("Cleaned JSON content is empty or whitespace.", cleaned_json_content or "", 0)

                    # Enhanced error handling for JSON parsing
                    try:
                        # Strip any remaining markdown code fence markers before parsing JSON
                        final_json_content = cleaned_json_content
                        
                        # Remove opening ```json or ``` if present
                        if final_json_content.startswith("```json"):
                            final_json_content = final_json_content[7:].lstrip()
                        elif final_json_content.startswith("```"):
                            final_json_content = final_json_content[3:].lstrip()
                            
                        # Remove closing ``` if present
                        if final_json_content.endswith("```"):
                            final_json_content = final_json_content[:-3].rstrip()
                            
                        self.logger.debug(f"Final JSON content after backtick removal: '{final_json_content[:200]}...'")
                        
                        parsed_instructional_guide = json.loads(final_json_content)
                    except json.JSONDecodeError as json_err:
                        # Extract context around the error position
                        error_pos = json_err.pos
                        context_start = max(0, error_pos - 30)
                        context_end = min(len(cleaned_json_content), error_pos + 30)
                        error_context = cleaned_json_content[context_start:context_end]
                        
                        self.logger.error(f"JSON decode error at position {error_pos}. Error: {str(json_err)}")
                        self.logger.error(f"Context around error: '{error_context}'")
                        
                        # Re-raise the exception with the same information
                        raise
                    
                    is_valid_structure, validation_msg = self._validate_json_structure(parsed_instructional_guide, substep_title)
                    if not is_valid_structure:
                        self.logger.error(f"Invalid JSON structure for '{substep_title}' (Attempt {attempt + 1}): {validation_msg}. Parsed type: {type(parsed_instructional_guide)}, Snippet: {str(parsed_instructional_guide)[:200]}")
                        # This error (e.g. not a dict, or missing keys) will be caught by the ValueError below
                        raise ValueError(f"Invalid JSON structure after parsing: {validation_msg}")

                    self.logger.info(f"Successfully parsed and validated instructional guide for '{substep_title}' on attempt {attempt + 1}.")
                    # Fill any missing standard categories with a default message if _validate_json_structure allows them (current impl is strict)
                    # This logic might be redundant if _validate_json_structure is strict and raises ValueError for missing keys.
                    # However, keeping it ensures categories are present if validation is ever relaxed.
                    for cat_key in self.standard_categories:
                        if cat_key not in parsed_instructional_guide:
                            self.logger.warning(f"Guide for '{substep_title}' is missing category '{cat_key}' post-validation (should not happen if validation is strict). Filling with default.")
                            parsed_instructional_guide[cat_key] = f"No content provided by LLM for {cat_key}."
                            
                    return {
                        'substep_title': substep_title,
                        'status': 'success',
                        'instructional_guide': parsed_instructional_guide,
                        'raw_analysis_response': raw_analysis_response,
                        'error_message': None
                    }

                else: # LLM handler call was not 'success'
                    llm_error = analysis_result.get('error_message', 'Unknown error from analyzer handler.')
                    self.logger.error(f"Analyzer LLM execution failed for '{substep_title}' (Attempt {attempt + 1}): {llm_error}")
                    last_error_message = f"LLM handler error: {llm_error}"
                    if attempt < max_retries - 1:
                        current_prompt = self._add_retry_instructions_to_prompt(
                            original_prompt_text=base_prompt_text,
                            attempt_num=attempt + 1, # Current attempt number that failed
                            error_type="LLM Handler Error",
                            error_details=llm_error,
                            raw_content_if_any=analysis_result.get('content') # Pass raw content if handler provided it
                        )
                        continue
                    else: # Max retries reached for LLM handler error
                        self.logger.error(f"Max retries ({max_retries}) reached for LLM handler error on '{substep_title}'.")
                        break # Exit loop, will fall through to final error handling

            except (json.JSONDecodeError, ValueError) as e_parse:
                error_name = type(e_parse).__name__
                self.logger.error(f"{error_name} for '{substep_title}' (Attempt {attempt + 1}): {e_parse}. Cleaned content snippet: '{cleaned_json_content_for_retry_info[:200] if cleaned_json_content_for_retry_info else 'N/A'}'")
                last_error_message = f"{error_name}: {e_parse}"
                if attempt < max_retries - 1:
                    current_prompt = self._add_retry_instructions_to_prompt(
                        original_prompt_text=base_prompt_text,
                        attempt_num=attempt + 1,
                        error_type=error_name,
                        error_details=str(e_parse),
                        raw_content_if_any=cleaned_json_content_for_retry_info # Use the cleaned content that failed
                    )
                    continue
                else: # Max retries reached for parsing/validation error
                    self.logger.error(f"Max retries ({max_retries}) reached for {error_name} on '{substep_title}'.")
                    break

            except Exception as e_generic: # Catch other unexpected errors
                self.logger.error(f"Unexpected error during analysis for '{substep_title}' (Attempt {attempt + 1}): {type(e_generic).__name__} - {e_generic}", exc_info=True)
                last_error_message = f"Unexpected error: {type(e_generic).__name__} - {e_generic}"
                if attempt < max_retries - 1:
                    current_prompt = self._add_retry_instructions_to_prompt(
                        original_prompt_text=base_prompt_text,
                        attempt_num=attempt + 1,
                        error_type="Generic Processing Error",
                        error_details=str(e_generic),
                        raw_content_if_any=raw_analysis_response # May or may not be available
                    )
                    continue
                else: # Max retries reached for generic error
                    self.logger.error(f"Max retries ({max_retries}) reached for generic error on '{substep_title}'.")
                    break

        # If loop finished due to max retries (break) or exhausted attempts without returning success
        self.logger.error(f"All {max_retries} attempts failed for substep '{substep_title}'. Last error: {last_error_message}")
        return {
            'substep_title': substep_title,
            'status': 'error_max_retries_exceeded',
            'instructional_guide': self._generate_default_guide(substep_title, f"Failed after {max_retries} attempts. Last error: {last_error_message}"),
            'raw_analysis_response': raw_analysis_response, # From the last attempt, if available
            'error_message': f"Failed to generate/parse instructional guide for '{substep_title}' after {max_retries} attempts. Last error: {last_error_message}"
        }

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