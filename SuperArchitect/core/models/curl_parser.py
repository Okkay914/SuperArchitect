import re
import shlex # Useful for basic shell-like splitting

def parse_curl(curl_command):
    """
    Parses a CURL command string to extract relevant parts for an HTTP request.
    Simplified version: focuses on URL, method (POST if -d), headers (-H), data (-d).
    """
    # Replace escaped quotes temporarily to help shlex, then restore them
    # This helps handle cases like -d '{"key": "value with \"quotes\""}'
    temp_placeholder_dq = "__TEMP_DOUBLE_QUOTE__"
    temp_placeholder_sq = "__TEMP_SINGLE_QUOTE__"
    curl_command = curl_command.replace('\\"', temp_placeholder_dq)
    curl_command = curl_command.replace("\\'", temp_placeholder_sq)

    try:
        parts = shlex.split(curl_command)
    except ValueError as e:
        # If shlex fails (e.g., unmatched quotes), raise a more specific error
        raise ValueError(f"Error splitting curl command (check quoting): {e}")

    # Restore escaped quotes
    parts = [part.replace(temp_placeholder_dq, '\\"').replace(temp_placeholder_sq, "\\'") for part in parts]

    if not parts or parts[0] != 'curl':
        raise ValueError("Not a valid curl command (must start with 'curl')")

    url = None
    method = 'GET' # Default
    headers = {}
    data = None
    explicit_method_set = False

    i = 1
    while i < len(parts):
        part = parts[i]
        if part in ('-H', '--header'):
            if i + 1 < len(parts):
                header_line = parts[i+1]
                if ':' in header_line:
                    key, value = header_line.split(':', 1)
                    headers[key.strip()] = value.strip()
                else:
                    # Handle headers without values if necessary, or raise error
                    # For now, assume key:value format
                    raise ValueError(f"Invalid header format (expected 'key: value'): {header_line}")
                i += 1 # Move past the value
            else:
                raise ValueError(f"Missing value for header option {part}")
        elif part in ('-d', '--data', '--data-raw', '--data-binary', '--data-urlencode'):
             if i + 1 < len(parts):
                data = parts[i+1]
                if not explicit_method_set: # Only set POST if method wasn't explicitly set by -X
                    method = 'POST'
                i += 1 # Move past the value
             else:
                raise ValueError(f"Missing value for data option {part}")
        elif part in ('-X', '--request'):
            if i + 1 < len(parts):
                method = parts[i+1].upper()
                explicit_method_set = True
                i += 1 # Move past the value
            else:
                 raise ValueError(f"Missing value for method option {part}")
        elif not part.startswith('-') and url is None:
            # Assume the first non-option part starting with http:// or https:// is the URL
            # This is a simplification; a real URL can be passed without options before it.
            # A more robust check might involve regex or urlparse.
            if part.startswith('http://') or part.startswith('https://'):
                 url = part
            # If it doesn't start with http/https, it might still be the URL if it's the last argument
            # or if subsequent arguments are clearly options. This basic parser assumes it's the first
            # non-option argument.
            elif i == len(parts) - 1 or parts[i+1].startswith('-'):
                 url = part # Assume it's the URL if it's the last part or followed by an option
            # Else, skip this part as it might be an unrecognized option or value

        # Add handling for other options like --user, -u if needed later
        # Example:
        # elif part in ('-u', '--user'):
        #     if i + 1 < len(parts):
        #         # Handle user:password, potentially encode to Basic Auth header
        #         user_pass = parts[i+1]
        #         # Add logic here, e.g., headers['Authorization'] = ...
        #         i += 1
        #     else:
        #         raise ValueError(f"Missing value for user option {part}")

        i += 1 # Move to the next part


    if url is None:
         # Try finding the last argument that looks like a URL if the simple logic failed
         for part in reversed(parts):
             if not part.startswith('-') and (part.startswith('http://') or part.startswith('https://')):
                 url = part
                 break
         if url is None:
            raise ValueError("Could not find URL in curl command")

    return {'url': url, 'method': method, 'headers': headers, 'data': data}

# Example Usage (for testing)
if __name__ == '__main__':
    test_curl_claude = "curl https://api.anthropic.com/v1/messages -H 'Content-Type: application/json' -H 'x-api-key: $ANTHROPIC_API_KEY' -H 'anthropic-version: 2023-06-01' -d '{\"model\": \"claude-3-opus-20240229\", \"max_tokens\": 4000, \"messages\": [{\"role\": \"user\", \"content\": \"$QUERY\"}]}'"
    test_curl_openai = "curl https://api.openai.com/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer $OPENAI_API_KEY' -d '{\"model\": \"gpt-4\", \"messages\": [{\"role\": \"user\", \"content\": \"$QUERY\"}]}'"
    test_curl_get = "curl https://example.com/data -H 'Accept: application/json'"
    test_curl_post_explicit = "curl -X POST https://example.com/submit -H 'Content-Type: text/plain' -d 'some data'"
    test_curl_with_escaped_quotes = "curl https://example.com/api -H 'Accept: application/json' -d '{\"text\": \"This has \\\"escaped quotes\\\" inside.\"}'"
    test_curl_no_protocol = "curl example.com/api -H 'Accept: application/json'" # Should ideally add https://

    tests = {
        "Claude": test_curl_claude,
        "OpenAI": test_curl_openai,
        "GET": test_curl_get,
        "POST Explicit": test_curl_post_explicit,
        "Escaped Quotes": test_curl_with_escaped_quotes,
        "No Protocol": test_curl_no_protocol,
    }

    for name, command in tests.items():
        print(f"--- Testing: {name} ---")
        print(f"Command: {command}")
        try:
            parsed = parse_curl(command)
            print("Parsed:", parsed)
            # Basic validation checks
            assert 'url' in parsed and parsed['url'] is not None
            assert 'method' in parsed
            assert 'headers' in parsed
            assert 'data' in parsed or parsed['method'] == 'GET' # Data can be None for GET
            print("Result: OK")
        except ValueError as e:
            print(f"Error parsing: {e}")
            print("Result: FAILED")
        print("-" * (len(name) + 14))