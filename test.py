# test.py
import torch
import torch.nn.functional as F
import torch.nn as nn

from dataset import causal_mask
from model import get_model
from config import get_latest_weights
from tokenizers import Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# NLTK zahteva preuzimanje resursa za METEOR
nltk.download('wordnet')
nltk.download('punkt')

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()

def compute_scores(reference_text, generated_text):
    # Tokenizacija za NLTK
    ref_tokens = nltk.word_tokenize(reference_text.lower())
    gen_tokens = nltk.word_tokenize(generated_text.lower())

    # BLEU Score (koristimo Smoothing jer su citati kratki)
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)

    # METEOR Score
    # Napomena: METEOR očekuje listu referenci kao stringove ili tokene u novijim verzijama
    m_score = meteor_score([ref_tokens], gen_tokens)

    return bleu, m_score

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


def run_validation_teacher_forcing(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Izračunaj loss
            loss = loss_function(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def run_validation_visualization(model, dataloader, src_tokenizer, tgt_tokenizer, device, num_examples=10):
    model.eval()
    bleu_scores = []
    meteor_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i == num_examples: break
            
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            target_text = batch['tgt_text'][0]
            single_encoder_input = encoder_input[0:1]
            single_encoder_mask = encoder_mask[0:1]
            
            # Generisanje (Greedy ili Sampling)
            model_out = greedy_decode(model, single_encoder_input, single_encoder_mask, tgt_tokenizer, device)
            if len(model_out.shape) > 1:
                model_out = model_out[0]

            model_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

            # Računanje skorova za ovaj primer
            bleu, meteor = compute_scores(target_text, model_text)
            bleu_scores.append(bleu)
            meteor_scores.append(meteor)
            
            if i < 3: # Ispiši samo prva 3 primera da ne zatrpaš konzolu
                print(f"EXPECTED: {target_text}")
                print(f"GENERATED: {model_text}")
                print(f"BLEU: {bleu:.4f} | METEOR: {meteor:.4f}\n")

    return sum(bleu_scores)/len(bleu_scores), sum(meteor_scores)/len(meteor_scores)

def greedy_decode(model, source, source_mask, tokenizer_tgt, device, max_len=96):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute encoder output
    encoder_output = model.encode(source, source_mask)
    # Start with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

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