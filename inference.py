import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, module="ssl")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import torch
from model import get_model
from config import get_config
from tokenizers import Tokenizer
import torch.nn.functional as F
from test import generate_quote 

def load_everything():
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    src_tokenizer = Tokenizer.from_file("tokenizer_src.json")
    tgt_tokenizer = Tokenizer.from_file("tokenizer_tgt.json")
    
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)

    model_path = "./weights/best_model.pt"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    return model, src_tokenizer, tgt_tokenizer, config, device

def run_inference(category_name, model, src_tokenizer, tgt_tokenizer, config, device):
    prompt = f"Generate a {category_name} quote:"
    
    with torch.no_grad():
        generated = generate_quote(
            model=model, 
            prompt=str(prompt),
            source_tokenizer=src_tokenizer, 
            target_tokenizer=tgt_tokenizer, 
            max_len=config['context_size'], 
            device=device, 
            top_k=50
        )
    
    print(f"\nQuote: {generated}")

if __name__ == "__main__":
    print("Loading model, please wait...")
    model, src_tokenizer, tgt_tokenizer, config, device = load_everything()
    
    print("\n--- Model is ready! ---")
    print("Type 'exit' to quit.")

    while True:
        cat = input("\nEnter category (love, life, motivational): ").strip()

        if cat.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        if not cat:
            continue

        run_inference(cat, model, src_tokenizer, tgt_tokenizer, config, device)