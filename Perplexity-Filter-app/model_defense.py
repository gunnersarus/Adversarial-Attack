import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

# model_defense.py
def load_model_and_tokenizer():
    
    llama7b = "huggyllama/llama-7b"
    guanaco7b = "timdettmers/guanaco-7b"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        llama7b,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={i: '24000MB' for i in range(torch.cuda.device_count())},
        force_download=False,
        use_cache=True
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, guanaco7b)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        llama7b,
        force_download=False,
        use_cache=True
    )
    
    return model, tokenizer

# Keep the original LLaMaWithPerplexityDefense class as is


class LLaMaWithPerplexityDefense:
    def __init__(
        self,
        device,
        model,
        tokenizer,
        perplexity_threshold=5.217528343200684,
        window_size=10
    ):
       
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        # Perplexity defense settings
        
        self.threshold = perplexity_threshold
        self.window_threshold = perplexity_threshold
        
        self.window_size = window_size
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def change_thresh(self,thresh):
        self.perplexity_threshold = thresh
    def change_window(self, window):
        self.window_size = window
    def change_model(self, model):
        self.model = model
    def change_tokenizer(self, tknz):
        self.tokenizer = tknz

    
    def calculate_perplexity(self, text):
        """Calculate the perplexity of the given text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, labels=inputs.input_ids)
            
        return outputs.loss.item()
    
    def calculate_window_perplexity(self, text):
        """Calculate perplexity for each window of tokens."""
        print(f"Threshold {self.window_threshold} window {self.window_size}")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        token_ids = inputs.input_ids
        
        window_perplexities = []
        
        for i in range(0, token_ids.size(1) - self.window_size + 1):
            window = token_ids[:, i:i + self.window_size]
            
            with torch.no_grad():
                outputs = self.model(input_ids=window, labels=window)
                window_perplexities.append(outputs.loss.item())
        
        return window_perplexities
    
    def is_sequence_safe(self, text):
        """Check if the sequence passes the perplexity defense checks."""
        # Calculate overall perplexity
        total_perplexity = self.calculate_perplexity(text)
        
        # Calculate window perplexities
        window_perplexities = self.calculate_window_perplexity(text)
        
        # Check if any window exceeds the threshold
        windows_safe = all(p <= self.window_threshold for p in window_perplexities)
        overall_safe = total_perplexity <= self.window_threshold
        
        return overall_safe and windows_safe, {
            'total_perplexity': total_perplexity,
            'window_perplexities': window_perplexities
        }
    

    def get_max_log_perplexity_of_goals(self, sequences):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        all_loss = []
        cal_log_prob = []
        for sequence in sequences:
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
            with torch.no_grad():   
                output = self.model(input_ids, labels=input_ids)
                loss = output.loss
            all_loss.append(loss.item())
            cal_log_prob.append(self.get_log_prob(sequence).mean().item())
        return max(cal_log_prob)
    
    def get_max_win_log_ppl_of_goals(self, sequences):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        all_loss = []
        for sequence in sequences:
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
            with torch.no_grad():   
                loss = self.model(input_ids, labels=input_ids).loss
            all_loss.append(loss.item())
        
        return max(all_loss)
    
    def get_log_prob(self, sequence):
        """
        Get the log probabilities of the token.

        Parameters
        ----------
        sequence : str
        """
        input_ids = self.tokenizer.encode(sequence, return_tensors='pt').cuda()
        with torch.no_grad():
            logits = self.model(input_ids, labels=input_ids).logits
        logits = logits[:, :-1, :].contiguous()
        input_ids = input_ids[:, 1:].contiguous()
        log_probs = self.cn_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        return log_probs
    
    def filter(self, sequences):
        """
        Filter sequences based on log perplexity.

        Parameters
        ----------
        sequences : list of str

        Returns
        -------
        filtered_log_ppl : list of float
            List of log perplexity values for each sequence.
        passed_filter : list of bool
            List of booleans indicating whether each sequence passed the filter.
        """
        filtered_log_ppl = []
        passed_filter = []
        for sequence in sequences:
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            if NLL_by_token.mean() <= self.threshold:
                passed_filter.append(True)
                filtered_log_ppl.append(NLL_by_token.mean().item())
            else:
                passed_filter.append(False)
                filtered_log_ppl.append(NLL_by_token.mean().item())
        return filtered_log_ppl, passed_filter
    
            
    def generate_safe_response(self, prompt, max_new_tokens=20):
            """Generate a response with perplexity defense checks."""
            # Check if prompt passes security checks
            is_safe, metrics = self.is_sequence_safe(prompt)
            print(metrics)
            if not is_safe:
                
                return "Evil and weird prompt. Ignore."
                
                
            # Format the prompt
            formatted_prompt = prompt
            
            # Generate response
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs=inputs.input_ids,
                max_new_tokens=max_new_tokens
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            
            return response