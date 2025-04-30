import os
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter # Added import

class FileManager:
    def __init__(self, base_data_path: str = 'SuperArchitect/data'):
        self.base_path = base_data_path
        self.queries_path = os.path.join(self.base_path, 'queries')
        self.responses_path = os.path.join(self.base_path, 'responses')
        self.analysis_path = os.path.join(self.base_path, 'analysis')
        self.final_path = os.path.join(self.base_path, 'final')
        self._ensure_directories()

    def _ensure_directories(self):
        """Creates the necessary data directories if they don't exist."""
        os.makedirs(self.queries_path, exist_ok=True)
        os.makedirs(self.responses_path, exist_ok=True)
        os.makedirs(self.analysis_path, exist_ok=True)
        os.makedirs(self.final_path, exist_ok=True)

    def _generate_run_id(self) -> str:
        """Generates a unique run ID based on the current timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_run(self, query: str, raw_responses: List[Dict[str, Any]], analysis_results: Dict[str, Any], final_report: str) -> str:
        """
        Saves all artifacts of a single search run to respective directories.
        Returns the generated run ID.
        """
        run_id = self._generate_run_id()
        print(f"Saving run with ID: {run_id}")

        try:
            # 1. Save Query
            query_filename = os.path.join(self.queries_path, f"run_{run_id}_query.txt")
            with open(query_filename, 'w', encoding='utf-8') as f:
                f.write(query)

            # 2. Save Raw Responses (as JSON)
            responses_filename = os.path.join(self.responses_path, f"run_{run_id}_responses.json")
            with open(responses_filename, 'w', encoding='utf-8') as f:
                json.dump(raw_responses, f, indent=2) # Directly dump raw_responses

            # 3. Save Analysis Results (as JSON)
            analysis_filename = os.path.join(self.analysis_path, f"run_{run_id}_analysis.json")
            # Convert Counter objects before saving if they exist
            serializable_analysis = self._make_serializable(analysis_results)
            with open(analysis_filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_analysis, f, indent=2)


            # 4. Save Final Report (as Markdown)
            report_filename = os.path.join(self.final_path, f"run_{run_id}_report.md")
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(final_report)

            print(f"Successfully saved artifacts for run {run_id}")
            return run_id

        except IOError as e:
            print(f"Error saving run {run_id}: {e}")
            # Handle error appropriately - maybe raise it or return an error indicator
            return f"ERROR_{run_id}"
        except TypeError as e:
             print(f"Error serializing data for run {run_id}: {e}")
             return f"ERROR_SERIALIZE_{run_id}"


    def _make_serializable(self, data: Any) -> Any:
         """Recursively converts Counter objects to dicts for JSON serialization."""
         if isinstance(data, dict):
             return {k: self._make_serializable(v) for k, v in data.items()}
         elif isinstance(data, list):
             return [self._make_serializable(item) for item in data]
         elif isinstance(data, Counter): # Check for Counter
             return dict(data) # Convert Counter to dict
         else:
             return data


# Example Usage (for testing within the file)
if __name__ == '__main__':
    # Assume we are running from the directory containing 'super_search'
    # Adjust path if necessary, e.g., FileManager('../data') if running from core
    fm = FileManager() # Uses default 'SuperArchitect/data'

    # Dummy data mimicking a run
    test_query = "What is the airspeed velocity of an unladen swallow?"
    test_responses = [
        {'status': 'success', 'model_name': 'ModelA', 'iteration': 1, 'content': 'African or European?'},
        {'status': 'error', 'model_name': 'ModelB', 'iteration': 1, 'error_message': 'Connection timeout'}
    ]
    test_analysis = {
        'total_requests': 2, 'successful_requests': 1, 'failed_requests': 1,
        'model_success_counts': Counter({'ModelA': 1}), # Use Counter here
        'keyword_tally': Counter({'african': 1, 'european': 1}), # Use Counter here
        'basic_summary': '...', 'errors': [{'model_name': 'ModelB', 'error_message': '...'}],
        'raw_successful_responses': [{'model_name': 'ModelA', 'content': '...'}]
    }
    test_report = "# Report\n## Summary\nSuccess: 1\n..."

    run_id = fm.save_run(test_query, test_responses, test_analysis, test_report)
    print(f"Test run saved with ID: {run_id}")

    # Verify files were created in SuperArchitect/data/... (manual check needed)