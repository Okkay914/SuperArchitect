import json
from typing import Dict, Any, Union, List # Add List if not already imported for mock_standard_categories
from core.analysis.engine import AnalyzerEngine
from core.models.base import ModelHandler # Assuming ModelHandler is in core.models.base

class MockModelHandler(ModelHandler):
    def __init__(self, model_name: str, api_key: str, config: Dict[str, Any]):
        super().__init__(api_key)
        self.model_name = model_name # Store model_name if needed by MockModelHandler
        self.config = config # Store config if needed by MockModelHandler

    def execute(self, prompt: str, is_json_output: bool = False) -> Union[str, Dict[str, Any]]:
        # This is a mock implementation.
        # The actual return value isn't critical for the JSON cleaning tests.
        if is_json_output:
            return {}
        return ""

# Mock objects for AnalyzerEngine instantiation
mock_config = {}
# Ensure the api_key provided results in a valid api_key_name for ModelHandler logic
# Use the new MockModelHandler here:
mock_analyzer_handler = MockModelHandler(model_name="test_mock_model", api_key="mock_api_key_gemini", config=mock_config)
mock_standard_categories: List[str] = ["Category1", "Category2"] # Ensure type hint for clarity

analyzer_engine_instance = AnalyzerEngine(
    config=mock_config,
    analyzer_handler=mock_analyzer_handler,
    standard_categories=mock_standard_categories
)

# Define diverse test case strings
test_cases = [
    # 1. Simple valid JSON string
    '{"key": "value", "number": 123}',

    # 2. JSON enclosed in markdown code fences
    """```json
{
  "name": "Test",
  "version": "1.0"
}
```""",

    # 3. JSON with trailing comma
    '{"fruit": "apple", "color": "red",}',

    # 4. JSON with single quotes for strings
    "{'item': 'book', 'price': 9.99}",

    # 5. JSON with missing comma (between key-value pairs)
    '{"city": "New York" "country": "USA"}',

    # 6. More complex, potentially malformed JSON string
    """{
    "data": {
        "items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        "status": "incomplete",,
        "notes": "Some notes here with 'quotes' and \\"escaped quotes\\" and a trailing comma somewhere maybe",
        "extra": null
    },
    "meta_info": "This part is outside but might be cleaned"
}""",

    # 7. JSON string with escaped characters
    '{"text": "This is a string with \\"quotes\\" and a newline\\ncharacter."}',

    # 8. JSON with leading/trailing whitespace
    '  {"whitespace": "test"}  ',

    # 9. JSON with just text around the JSON
    "Some leading text before the JSON {\"leading_text_test\": true} and some trailing text.",

    # 10. Empty JSON object
    "{}",

    # 11. Empty JSON array
    "[]",

    # 12. JSON with mixed quotes that might need fixing
    '{"name": "John Doe", \'age\': 30, "city": \'New York\'}',

    # 13. JSON with comments (should be stripped)
    """{
        // This is a single line comment
        "comment_test": "value1",
        /* This is a
           multi-line comment */
        "another_key": "value2"
    }""",

    # 14. JSON with unescaped newlines within strings (should be fixed)
    '{"message": "Hello\nWorld"}',

    # 15. A string that is definitively not JSON
    "This is just a plain sentence and not JSON.",

    # 16. JSON with boolean/null values written as strings (should be converted to literals)
    '{"is_active": "true", "has_data": "false", "optional_field": "null"}',

    # 17. More deeply nested structure with mixed issues
    """{
        'id': 'user123', // User ID
        "profile": {
            "name": "Alice Wonderland",
            "email": 'alice@example.com'
            "details": {
                "is_verified": "true",, // Double comma here
                "preferences": [
                    {'theme': 'dark', "notifications": "false"},
                    {'lang': 'en' "font": "Arial"} // Missing comma in array
                ],
                "notes": "Line one.\nLine two with 'single' quotes."
            }
        },
        "status": "active",
        "last_login": None, // Python None
        "enabled": True // Python True
    }"""
]

def run_tests():
    for i, test_str in enumerate(test_cases):
        print(f"--- Test Case {i+1} ---")
        print(f"Original String:\n{test_str}\n")

        cleaned_str = analyzer_engine_instance._clean_llm_json_output(test_str)
        print(f"Cleaned String:\n{cleaned_str}\n")

        try:
            parsed_json = json.loads(cleaned_str)
            print("Parsing Successful:")
            print(json.dumps(parsed_json, indent=2))
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
        print("-" * 20 + "\n")

if __name__ == "__main__":
    run_tests()