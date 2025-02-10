import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_response(user_query):
    """Fetch response from Azure OpenAI GPT model."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_query}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"
