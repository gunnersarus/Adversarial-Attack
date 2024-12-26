# app.py
from flask import Flask, render_template, request, jsonify
import torch
from fastchat.model import load_model, get_conversation_template
import os

app = Flask(__name__)

# Khởi tạo model khi start app
def init_model():
    # Giả định các tham số model, có thể điều chỉnh theo nhu cầu
    class ModelArgs:
        def __init__(self):
            self.model_path = "lmsys/vicuna-7b-v1.1"  # Thay đổi path tới model của bạn
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.num_gpus = 1
            self.max_gpu_memory = None
            self.load_8bit = False
            self.cpu_offloading = False
            self.revision = None
            self.debug = False
            self.temperature = 0.7
            self.repetition_penalty = 1.0
            self.max_new_tokens = 1024

    args = ModelArgs()
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    return model, tokenizer, args

model, tokenizer, model_args = init_model()

def paraphrase_text(text, enable_paraphrase=False, temperature=0.7):
    """Xử lý văn bản với model"""
    try:
        if enable_paraphrase:
            # Tạo prompt cho paraphrase
            paraphrase_prompt = "Please rephrase the following text while maintaining its meaning: "
            conv = get_conversation_template(model_args.model_path)
            conv.append_message(conv.roles[0], paraphrase_prompt + text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Generate paraphrase
            inputs = tokenizer([prompt], return_tensors="pt").to(model_args.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True if temperature > 1e-5 else False,
                    temperature=0.2,
                    repetition_penalty=1,
                    max_new_tokens=100,
                )
            
            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs["input_ids"][0]):]
            
            paraphrased = tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            
            # Process paraphrased text
            text_to_process = paraphrased
        else:
            text_to_process = text
            paraphrased = None
        
        # Final processing
        conv = get_conversation_template(model_args.model_path)
        conv.append_message(conv.roles[0], text_to_process)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        inputs = tokenizer([prompt], return_tensors="pt").to(model_args.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True if temperature > 1e-5 else False,
                temperature=temperature,
                repetition_penalty=model_args.repetition_penalty,
                max_new_tokens=model_args.max_new_tokens,
            )
        
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
        
        final_output = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        
        return {
            "success": True,
            "paraphrased": paraphrased,
            "final_output": final_output
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    text = data.get('text', '')
    enable_paraphrase = data.get('enable_paraphrase', False)
    temperature = float(data.get('temperature', 0.5))
    
    result = paraphrase_text(text, enable_paraphrase, temperature)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)