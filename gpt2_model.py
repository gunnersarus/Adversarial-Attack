import gradio as gr
import requests

# API URL and header with your token
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": "Bearer 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"}  # Use your token



def query(payload):
    # Send a POST request to the API
    response = requests.post(API_URL, headers=headers, json=payload)
    # Check if the response status code is not 200 (OK)
    if response.status_code != 200:
        return {"error": f"Request failed with status code: {response.status_code}"}
    return response.json()  # Return the JSON response

def generate_response(user_input):
    # Send request to the API
    output = query({"inputs": user_input})

    # Extract generated text from the response
    if isinstance(output, list) and len(output) > 0:
        gpt_output = output[0]['generated_text']
        return gpt_output  # Return the generated output
    else:
        return "No response from the model."  # Return a default message if there's no valid response

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs="text",  # Input type
    outputs="text",  # Output type
    title="GPT-2 Interface",  # Title of the interface
    description="Enter a piece of text to generate a response from GPT-2."  # Description of the interface
)

# Launch the interface with sharing option
if __name__ == "__main__":
    iface.launch(share=True)
