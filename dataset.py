from pathlib import Path
import pandas as pd
import torch

from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer

from typing import Any, Dict

class QuoteDataset(TorchDataset):
    def __init__(self, dataset: HFDataset, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, src_lang: str, tgt_lang: str, context_size: int) -> None:
        super().__init__()
        self.context_size = context_size
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id('[PAD]')], dtype = torch.int64)


    def __len__(self) -> int:
        return len(self.dataset)
    
    
    def __getitem__( self, index: int) -> Dict[str, Any]:
        item = self.dataset[index]
        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        enc_input_tokens = self.src_tokenizer.encode(src_text).ids[:self.context_size - 2]
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids[:self.context_size - 1]

        enc_num_padding_tokens = self.context_size - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.context_size - len(dec_input_tokens) - 1

        # Encoder input: [SOS] + tokens + [EOS] + [PAD]...
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(enc_num_padding_tokens)
        ])

        # Decoder input: [SOS] + tokens + [PAD]...
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.pad_token.repeat(dec_num_padding_tokens)
        ])

        # Label: tokens + [EOS] + [PAD]...
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(dec_num_padding_tokens)
        ])

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

    
def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0

def load_data_quotes(csv_file: str, src_lang: str, tgt_lang: str, config: dict, sample_size=None):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['quote', 'author', 'category'])
    def get_first_tag(cat):
        try:
            import ast
            tags = ast.literal_eval(cat) if isinstance(cat, str) and cat.startswith('[') else cat
            if isinstance(tags, list) and len(tags) > 0:
                return str(tags[0])
            return str(tags)
        except:
            return str(cat)

    df['category'] = df['category'].apply(get_first_tag)
    df = df.drop_duplicates(subset=['quote'])
    if sample_size and len(df) > sample_size:
        oversample = int(sample_size * 1.2)
        df = df.sample(n=min(oversample, len(df)), random_state=config.get('seed', 42))

    df['prompt'] = "Generate a " + df['category'] + " quote:"
    
    formatted_data = [
        {
            "id": str(i),
            "translation": {src_lang: row['prompt'], tgt_lang: row['quote']}
        }
        for i, row in df.iterrows()
    ]

    dataset = HFDataset.from_list(formatted_data)
    return dataset