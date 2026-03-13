# test.py
import torch
import torch.nn.functional as F
import torch.nn as nn

from dataset import causal_mask
from model import get_model
from config import get_latest_weights
from tokenizers import Tokenizer

def load_model_and_tokenizers(config, device='cpu'):

    # Tokenizers
    src_tokenizer = Tokenizer.from_file(config['tokenizer_file'].format(config['source_language']))
    tgt_tokenizer = Tokenizer.from_file(config['tokenizer_file'].format(config['target_language']))

    # Model
    vocab_src = src_tokenizer.get_vocab_size()
    vocab_tgt = tgt_tokenizer.get_vocab_size()
    model = get_model(config, vocab_src, vocab_tgt).to(device)

    # Preload weights
    if config['preload']:
        model_path = get_latest_weights(config) if config['preload'] == 'latest' else f"{config['model_folder']}/{config['model_basename']}{config['preload']}.pt"
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    
    model.eval()
    return model, src_tokenizer, tgt_tokenizer


def run_validation(model, dataloader, loss_function, tokenizer, device, epoch, src_tokenizer):
    # Na kraju epoch
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            transformer_output = model.project(decoder_output)

            loss = loss_function(transformer_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Validation loss after epoch {epoch}: {avg_loss:.3f}")
    prompt = "Generate a dark quote:"
    quote = generate_quote(model, src_tokenizer, tokenizer, prompt, max_len=64, device=device, top_k=50, temperature=1.0)

    print("Sample quote:", quote)
    return avg_loss


def run_validation_teacher_forcing(*args, **kwargs):
    return

def run_validation_visualization(*args, **kwargs):
    return

def run_test(model, test_dataloader, src_tokenizer, tgt_tokenizer, device):
    model.eval()
    loss_function = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id('[PAD]'))
    total_loss = 0

    # Evaluate on full test set
    with torch.no_grad():
        for batch in test_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            transformer_output = model.project(decoder_output)

            loss = loss_function(transformer_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"Average test loss: {avg_loss:.3f}\n")

    # Generate sample quotes from different prompts
    prompts = [
        "Generate a dark quote:",
        "Generate a love quote:",
        "Generate a motivational quote:",
        "Generate a life quote:",
        "Generate a family quote:"
    ]

    for prompt in prompts:
        quote = generate_quote(model, src_tokenizer, tgt_tokenizer, prompt, max_len=64, device=device, top_k=50, temperature=1.0)
        print(f"Prompt: {prompt}")
        print(f"Generated: {quote}\n")



def generate_quote(model, source_tokenizer, target_tokenizer, prompt, 
                   max_len=64, device='cpu', top_k=50, temperature=1.0):
    """
    Generate a quote from the model using top-k sampling.

    Args:
        model: Transformer model.
        source_tokenizer: Tokenizer for input.
        target_tokenizer: Tokenizer for output.
        prompt (str): Text prompt.
        max_len (int): Maximum length of generated sequence.
        device (str): 'cpu' or 'cuda'.
        top_k (int): Number of top tokens to sample from.
        temperature (float): Sampling temperature.

    Returns:
        str: Generated text.
    """
    model.eval()
    with torch.no_grad():
        # --- Encode prompt ---
        input_ids = source_tokenizer.encode(prompt).ids
        input_tensor = torch.tensor(
            [source_tokenizer.token_to_id('[SOS]')] + input_ids + [source_tokenizer.token_to_id('[EOS]')],
            dtype=torch.int64
        ).unsqueeze(0).to(device)
        encoder_mask = (input_tensor != source_tokenizer.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).int()

        encoder_output = model.encode(input_tensor, encoder_mask)

        # --- Decoder loop ---
        decoder_ids = [target_tokenizer.token_to_id('[SOS]')]

        for _ in range(max_len):
            decoder_input = torch.tensor([decoder_ids], dtype=torch.int64).to(device)
            mask = causal_mask(len(decoder_ids)).to(device)
            decoder_mask = (decoder_input != target_tokenizer.token_to_id('[PAD]')).unsqueeze(1).int() & mask

            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            logits = model.project(decoder_output)  # (1, seq_len, vocab_size)
            logits_last = logits[0, -1] / temperature  # apply temperature

            # --- Top-k sampling ---
            if top_k > 0:
                topk_probs, topk_indices = torch.topk(F.softmax(logits_last, dim=-1), top_k)
                topk_probs = topk_probs / topk_probs.sum()  # normalize
                next_token_id = topk_indices[torch.multinomial(topk_probs, 1)].item()
            else:
                # Greedy fallback
                next_token_id = torch.argmax(logits_last).item()

            if next_token_id == target_tokenizer.token_to_id('[EOS]'):
                break

            decoder_ids.append(next_token_id)

        # --- Decode tokens to string ---
        generated_text = target_tokenizer.decode(decoder_ids[1:])  # skip SOS
        return generated_text

# def generate_quote(model, source_tokenizer, target_tokenizer, prompt, max_len=64, device='cpu'):
#     model.eval()
#     with torch.no_grad():
#         # Tokenizuj prompt
#         input_ids = source_tokenizer.encode(prompt).ids
#         input_tensor = torch.tensor([source_tokenizer.token_to_id('[SOS]')] + input_ids + [source_tokenizer.token_to_id('[EOS]')], dtype=torch.int64).unsqueeze(0).to(device)
#         encoder_mask = (input_tensor != source_tokenizer.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).int()

#         # Encode
#         encoder_output = model.encode(input_tensor, encoder_mask)

#         # Start decoder with SOS
#         decoder_ids = [target_tokenizer.token_to_id('[SOS]')]
#         for _ in range(max_len):
#             decoder_input = torch.tensor([decoder_ids], dtype=torch.int64).to(device)
#             mask = causal_mask(len(decoder_ids)).to(decoder_input.device)
#             decoder_mask = (decoder_input != target_tokenizer.token_to_id('[PAD]')).unsqueeze(1).int() & mask

#            # decoder_mask = (decoder_input != target_tokenizer.token_to_id('[PAD]')).unsqueeze(1).int() & causal_mask(len(decoder_ids)))
            
#             decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
#             logits = model.project(decoder_output)
#             next_token_id = torch.argmax(logits[0, -1]).item()
            
#             if next_token_id == target_tokenizer.token_to_id('[EOS]'):
#                 break
#             decoder_ids.append(next_token_id)

#         # Dekoduj u string
#         generated_text = target_tokenizer.decode(decoder_ids[1:])  # skip SOS
#         return generated_text

