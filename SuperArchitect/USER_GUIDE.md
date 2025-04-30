# Super Search User Guide

This guide explains how to set up and use the Super Search tool.

## What is Super Search?

Super Search helps you research topics by using advanced AI models to analyze information and provide a synthesized answer. You give it a question, and it uses different AI services to find and combine information to give you a comprehensive response.

## Getting Started: API Keys

To use Super Search, you need special "keys" (like passwords) to connect to the different AI services it uses. Think of these keys as permission slips that allow Super Search to ask the AI for help.

The main services used are:

*   **OpenAI:** Provides powerful AI models like GPT.
*   **Anthropic Claude:** Another provider of advanced AI models.
*   **Google Gemini:** Google's family of AI models.

**How to get your keys:**

1.  **Sign Up:** You'll need to create accounts with each service you want to use. You can usually find their websites by searching for "OpenAI API key", "Anthropic Claude API key", and "Google Gemini API key".
2.  **Find the API Key Section:** Once logged in, look for a section related to "API Keys", "Developer Settings", or similar.
3.  **Create a New Key:** Follow the instructions on each site to generate a new API key.
4.  **Copy Your Key:** Copy the generated key immediately. It might only be shown once!

**IMPORTANT:** Your API keys are like passwords for these powerful AI services. **NEVER share them with anyone or post them publicly.** Keep them safe!

## Configuration: Setting up `config.yaml`

Super Search needs a configuration file named `config.yaml` to know your API keys and which AI models you want to use.

1.  **Find the Example:** Look for a file named `config.example.yaml` in the `super_search` directory.
2.  **Make a Copy:** Create a copy of `config.example.yaml` in the *same* directory.
3.  **Rename the Copy:** Rename the copied file to exactly `config.yaml`.

Now, open `config.yaml` with a simple text editor. You'll see sections like this:

```yaml
# API Keys for different services
api_keys:
  openai: YOUR_API_KEY_HERE
  claude: YOUR_API_KEY_HERE
  gemini: YOUR_API_KEY_HERE

# Model selection
models:
  # Which AI reads and summarizes the initial information?
  # Options: 'openai', 'claude', 'gemini'
  summarizing_model: openai

  # Which AI writes the final answer based on the summaries?
  # Options: 'openai', 'claude', 'gemini'
  final_architect_model: claude
```

**Setting your keys:**

*   Find the `api_keys:` section.
*   Replace `YOUR_API_KEY_HERE` next to `openai:` with the actual API key you got from OpenAI.
*   Do the same for `claude:` and `gemini:` with their respective keys.
*   If you don't have a key for a service, you can leave it as `YOUR_API_KEY_HERE` or delete that line, but the tool might not work if the selected model requires that key.

**Choosing your models:**

*   Under the `models:` section:
    *   `summarizing_model`: Choose which AI service you want to use for the first step of reading and summarizing information. Change `openai` to `claude` or `gemini` if you prefer.
    *   `final_architect_model`: Choose which AI service you want to write the final combined answer. You can use the same AI as the summarizer or a different one. Change `claude` to `openai` or `gemini` if you prefer.

**Save the file.** Remember, this `config.yaml` file now contains your private keys, so keep it secure and don't share it!

## Running Super Search

Super Search is run from a command line interface (like Terminal on Mac/Linux or Command Prompt/PowerShell on Windows).

1.  **Open your terminal.**
2.  **Navigate (if necessary):** You might need to use `cd` commands to get to the directory *above* `super_search`.
3.  **Run the command:** Type the following, replacing the example question with your own:

    ```bash
    python super_search/main.py "Your research question goes here in quotes"
    ```

    *Example:*
    ```bash
    python super_search/main.py "What were the main causes of the French Revolution?"
    ```

4.  **Wait for the answer:** The tool will start working, using the models you configured. The final answer will be printed directly into your terminal window. This might take a little while depending on the complexity of your question.

That's it! You can now use Super Search for your research.