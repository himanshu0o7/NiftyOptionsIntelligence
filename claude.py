"""
Claude AI integration with proper error handling for live market environment.
"""
import os
import logging

logger = logging.getLogger(__name__)

try:
    import anthropic
    
    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        client = None
    else:
        client = anthropic.Anthropic(api_key=api_key)
        
except ImportError:
    logger.warning("anthropic package not installed. Claude functionality will be disabled.")
    client = None


def ask_claude(prompt: str) -> str:
    """Ask Claude AI a question with proper error handling."""
    if client is None:
        return "Claude AI is not available. Please install anthropic package and set ANTHROPIC_API_KEY."
    
    try:
        response = client.messages.create(
            model="claude-3.5-sonnet-20241022", 
            max_tokens=1000, 
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude AI request failed: {e}")
        return f"Error: {e}"


# Example usage (commented out for production)
# if __name__ == "__main__":
#     result = ask_claude("Write a Python script to fetch BTC/INR price from CoinSwitch API")
#     print(result)

