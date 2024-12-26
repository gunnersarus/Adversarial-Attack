import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "huggyllama/llama-7b"
adapters_name = 'timdettmers/guanaco-7b'

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization (or set load_in_8bit=True if preferred)
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load the model with quantization_config (don't use load_in_4bit here)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,  # Use quantization_config without load_in_4bit
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={i: '24000MB' for i in range(torch.cuda.device_count())},
    force_download=False,
    use_cache=True
)

# Load the adapter model
model = PeftModel.from_pretrained(model, adapters_name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True,  use_cache=True)

# Prepare the prompt
prompt = "what is the capital of vietnam"
formatted_prompt = (
    f"A chat between a curious human and an artificial intelligence assistant."
    f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    f"### Human: {prompt} ### Assistant:"
)

# Tokenize and generate the output
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=20)

# Decode and print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
