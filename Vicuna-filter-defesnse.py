import argparse
import logging
from typing import Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from fastchat.model import load_model, get_conversation_template, add_model_args
from fastchat.serve.inference import generate_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerplexityFilter:
    """Filter text based on perplexity scores."""
    
    def __init__(self, model: torch.nn.Module, tokenizer, threshold: float = 5.0):
        self.tokenizer = tokenizer
        self.model = model
        self.threshold = threshold
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        
    def calculate_perplexity(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        
        return logits, labels
    
    def filter(self, sequence: str) -> Tuple[bool, float]:
        try:
            logits, labels = self.calculate_perplexity(sequence)
            log_probs = self.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            perplexity = log_probs.mean().item()
            
            return perplexity <= self.threshold, perplexity
            
        except Exception as e:
            logger.error(f"Error in perplexity calculation: {str(e)}")
            return True, 0.0

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    response: str
    perplexity_score: float

def create_app(
    model: torch.nn.Module,
    tokenizer,
    ppl_threshold: float,
    conversation_template: str = "vicuna_v1.1"
) -> FastAPI:
    app = FastAPI(
        title="Custom Language Model API",
        description="API for text generation with perplexity filtering",
        version="1.0.0"
    )
    
    ppl_filter = PerplexityFilter(model, tokenizer, ppl_threshold)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> Response:
        # Check perplexity
        passed_filter, ppl_score = ppl_filter.filter(request.prompt)
        
        if not passed_filter:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Input rejected due to high perplexity",
                    "threshold": ppl_threshold,
                    "score": ppl_score
                }
            )

        # Set up conversation
        conv = get_conversation_template(conversation_template)
        conv.append_message(conv.roles[0], request.prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Generate response
        try:
            chunks = []
            for chunk in generate_stream(
                model=model,
                tokenizer=tokenizer,
                params={
                    "prompt": prompt,
                    "temperature": request.temperature,
                    "max_new_tokens": request.max_new_tokens,
                    "repeat_penalty": request.repetition_penalty,
                    "stop": None,
                    "stop_token_ids": None,
                    "echo": False
                },
                device=next(model.parameters()).device,
                context_len=2048,
                stream_interval=2
            ):
                # Extract the text from the chunk
                if isinstance(chunk, dict) and "text" in chunk:
                    chunks.append(chunk["text"])
                elif isinstance(chunk, str):
                    chunks.append(chunk)
                else:
                    logger.warning(f"Unexpected chunk format: {chunk}")
            
            response = "".join(chunks)
            
            return GenerateResponse(
                response=response,
                perplexity_score=ppl_score
            )
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating response: {str(e)}"
            )

    return app

def main():
    parser = argparse.ArgumentParser(description="Run the custom language model API server")
    add_model_args(parser)
    parser.add_argument("--ppl_threshold", type=float, default=5.0,
                       help="Threshold for perplexity filtering")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to run the server on")
    
    args = parser.parse_args()

    logger.info("Loading model...")
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision
    )
    
    logger.info(f"Starting server with perplexity threshold: {args.ppl_threshold}")
    app = create_app(model, tokenizer, args.ppl_threshold)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()