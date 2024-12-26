# app.py
import torch
from flask import Flask, render_template, request, jsonify
from model_defense import LLaMaWithPerplexityDefense, load_model_and_tokenizer

app = Flask(__name__)

# Initialize model and defense system
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer()
defended_model = LLaMaWithPerplexityDefense(
    model=model,
    tokenizer=tokenizer,
    device=device,
    perplexity_threshold=5.217528343200684,
    window_size=10
)

# Global variable to track defense status
defense_enabled = True

@app.route('/')
def home():
    return render_template('index.html', defense_status=defense_enabled)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 100))
    
    if not defense_enabled:
        # Generate response without defense
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs=inputs.input_ids,
            max_new_tokens=max_tokens
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        safety_info = {"defense_active": False}
    else:
        # Generate response with defense
        is_safe, metrics = defended_model.is_sequence_safe(prompt)
        if not is_safe:
            response = "⚠️ Prompt rejected by safety filter"
        else:
            response = defended_model.generate_safe_response(prompt, max_new_tokens=max_tokens)
        safety_info = {
            "defense_active": True,
            "is_safe": is_safe,
            "metrics": metrics
        }
    
    return jsonify({
        "response": response,
        "safety_info": safety_info
    })

@app.route('/toggle_defense', methods=['POST'])
def toggle_defense():
    global defense_enabled
    defense_enabled = not defense_enabled
    return jsonify({"defense_enabled": defense_enabled})

if __name__ == '__main__':
    app.run(debug=True)
