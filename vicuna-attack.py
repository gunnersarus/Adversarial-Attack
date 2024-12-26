import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api(prompt: str, url: str = "http://localhost:8000/generate"):
    """
    Send a single request to the FastChat API.
    
    Args:
        prompt: The text prompt to send
        url: The API endpoint URL
    """
    logger.info(f"Sending request with prompt: {prompt}")
    
    try:
        response = requests.post(
            url,
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_new_tokens": 512,
                "repetition_penalty": 1.0
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Perplexity score: {result['perplexity_score']:.2f}")
            logger.info(f"Response: {result['response']}")
        else:
            logger.error(f"Error: {response.json()['detail']}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    # You can change this prompt as needed
    test_prompt = "Hello"  # This should trigger the perplexity filter
    test_api(test_prompt)