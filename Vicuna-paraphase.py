import argparse
import torch
from fastchat.model import load_model, get_conversation_template, add_model_args
from tenacity import retry, stop_after_attempt, wait_random_exponential

def paraphrase_input(model, tokenizer, original_text, args, prompt=None):
    """
    Paraphrase input text using Vicuna model
    """
    assert prompt, "Prompt must be provided for paraphrasing"
    
    # Combine prompt with original text
    paraphrase_query = prompt + original_text
    
    # Create conversation template
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], paraphrase_query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Retry mechanism for robust inference
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
    def generate_with_backoff(input_prompt):
        inputs = tokenizer([input_prompt], return_tensors="pt").to(args.device)
        output_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
        
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]):]
            
        return tokenizer.decode(
            output_ids, 
            skip_special_tokens=True, 
            spaces_between_special_tokens=False
        )

    paraphrased_text = generate_with_backoff(prompt)
    return paraphrased_text

@torch.inference_mode()
def main(args):
    # Load model
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

    # Original message
    original_msg = args.message
    print(f"Original message: {original_msg}\n")

    # Paraphrase the input if paraphrasing is enabled
    if args.enable_paraphrase:
        paraphrase_prompt = "Please rephrase the following text while maintaining its meaning: "
        paraphrased_msg = paraphrase_input(
            model,
            tokenizer,
            original_msg,
            args,
            prompt=paraphrase_prompt
        )
        msg_to_process = paraphrased_msg
        print(f"Paraphrased message: {paraphrased_msg}\n")
    else:
        msg_to_process = original_msg

    # Process the message (original or paraphrased)
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg_to_process)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True if args.temperature > 1e-5 else False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    # Print final results
    print(f"Final response:")
    print(f"{conv.roles[0]}: {msg_to_process}")
    print(f"{conv.roles[1]}: {outputs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable-paraphrase", action="store_true", 
                       help="Enable input paraphrasing before processing")
    parser.add_argument("--message", type=str, required=True,
                       help="Input message to process")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)