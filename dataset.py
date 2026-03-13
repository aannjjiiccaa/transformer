from pathlib import Path
import pandas as pd
import torch

from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from translate.storage.tmx import tmxfile

from typing import Any, Dict

class BilingualDataset(TorchDataset):
    """
    Wrapper class of Torch Dataset.
    Has to have methods __init__, __len__ and __getitem__ to function properly.
    """

    def __init__(
            self, 
            dataset: HFDataset, 
            source_tokenizer: Tokenizer, 
            target_tokenizer: Tokenizer, 
            source_language: str, 
            target_language: str, 
            context_size: int
        ) -> None:
        """Initializing the BilingualDataset object.

        Args:
            dataset (HFDataset): 
                HuggingFace dataset with columns id and translations.
                    id is the number of the current row.
                    translations is a dictionary with entries 'language': sentence,
                    which has at least two different languages.
            source_tokenizer (Tokenizer): Tokenizer for the source language.
            target_tokenizer (Tokenizer): Tokenizer for the target language.
            source_language (str): Source language for the translations.
            target_language (str): Target language for the translations.
            context_size (int): Maximum allowed length of a sentence (in either language).
        """
        super().__init__()

        # Initializing context size.
        self.context_size = context_size

        # Initializing the dataset.
        self.dataset = dataset

        # Initializing the tokenizers.
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        # Initializing the languages.
        self.source_language = source_language
        self.target_language = target_language

        # Initializing the start of sentence, end of sentence and padding tokens.
        
        # Start of sentence token signifies the beginning of a sentence.
        self.sos_token = torch.tensor([source_tokenizer.token_to_id('[SOS]')], dtype = torch.int64)
        
        # End of sentence token signifies the end of a sentence.
        self.eos_token = torch.tensor([source_tokenizer.token_to_id('[EOS]')], dtype = torch.int64)

        # Padding token signifies the placeholder token for sentences shorter than context size, which fills the empty spaces.
        self.pad_token = torch.tensor([source_tokenizer.token_to_id('[PAD]')], dtype = torch.int64)


    def __len__(self) -> int:
        """
        Returns:
            int: Number of sentences in the dataset.
        """
        return len(self.dataset)
    
    
    def __getitem__(
            self, 
            index: int
        ) -> Dict[str, Any]:
        """Gets the row from the dictionary at a specified index.

        Args:
            index (int): Index at which to return the element from the list.

        Raises:
            ValueError: _description_

        Returns:
            Dict[str, Any]: A dictionary with 7 fields:
                encoder_input: 
                    Input to be fed to the encoder. 
                    Tensor of dimension (context_size)
                decoder_input:
                    Input to be fed to the decoder. 
                    Tensor of dimension (context_size)
                encoder_mask:
                    Mask for the encoder, that will mask any padding tokens.
                    Tensor of dimension (1, 1, context_size)
                decoder_mask:
                    Mask for the decoder, that will mask any padding tokens and won't allow predictions in the past.
                    Tensor of dimension (1, context_size, context_size)
                label:
                    Expected model output.
                    Tensor of dimension (context_size)
                source_text:
                    Sentence in the source language.
                target_text:
                    Sentence in the target language.
        """
        # Get the index-th row of the dataset.
        source_target_pair = self.dataset[index]

        # Get the sentence in the source and target language.
        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]

        # Number of tokens in sentences.
        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids

        # Number of padding tokens for both sentences.
        # Encoder already has len(encoder_input_tokens), SOS and EOS.
        encoder_num_padding_tokens = self.context_size - len(encoder_input_tokens) - 2
        # Decoder already has len(decoder_input_tokens), and:
        #       SOS token for the input;
        #       EOS token for the label.
        decoder_num_padding_tokens = self.context_size - len(decoder_input_tokens) - 1
        
        # Make sure the sentence isn't too long in either language.
        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long!")
        
        # Encoder input is [SOS] token_enc[1] token_enc[2] ... token_enc[K] [EOS] [PAD] [PAD] ... [PAD].
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Decoder input is [SOS] token_dec[1] token_dec[2] ... token_dec[J] [PAD] [PAD] ... [PAD].
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Label is token_dec[1] token_dec[2] ... token_dec[J] [EOS] [PAD] [PAD] ... [PAD].
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Make sure the tensor dimensions are correct.
        assert encoder_input.size(0) == self.context_size
        assert decoder_input.size(0) == self.context_size
        assert label.size(0) == self.context_size

        # Return the appropriate values.
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "source_text" : source_text,
            "target_text" : target_text
        }

    
def causal_mask(size: int) -> torch.Tensor:
    """
    Generates a causal mask for the decoder. This is a triangular matrix that
    has all ones as inputs which deals with decoder having access to words that
    have not yet been translated.

    Args:
        size (int): Size of the mask matrix.

    Returns:
        torch.Tensor: Triangular matrix with all ones, of dimension (1, size, size).
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0

def get_first_tag(cat):
    if isinstance(cat, list) and len(cat) > 0:
        return cat[0]
    if isinstance(cat, str):
        try:
            import ast
            tags = ast.literal_eval(cat)
            return tags[0] if tags else "general"
        except:
            return cat
    return "general"

def load_data_quotes(csv_file: str, src_lang: str, tgt_lang: str, config: dict, sample_size=None):
    """
    Učitava citate, čisti bazu i priprema prompte, 
    ali ostavlja precizno filtriranje za kasnije (kad imamo tokenizer).
    """
    df = pd.read_csv(csv_file)
    
    # 1. Osnovno čišćenje (uklanjanje praznih polja)
    df = df.dropna(subset=['quote', 'author', 'category'])
    
    # 2. Sređivanje kategorija (uzimamo samo prvi tag)
    def get_first_tag(cat):
        try:
            import ast
            # Ako je string u formatu "['love', 'life']", pretvori u listu
            tags = ast.literal_eval(cat) if isinstance(cat, str) and cat.startswith('[') else cat
            if isinstance(tags, list) and len(tags) > 0:
                return str(tags[0])
            return str(tags)
        except:
            return str(cat)

    df['category'] = df['category'].apply(get_first_tag)
    
    # 3. Uklanjanje duplikata da model ne uči napamet
    df = df.drop_duplicates(subset=['quote'])

    # 4. Sampling (uzimamo više jer ćemo posle filtrirati)
    if sample_size and len(df) > sample_size:
        # Uzmi 20% više od sample_size jer će filtriranje tokena izbaciti neke redove
        oversample = int(sample_size * 1.2)
        df = df.sample(n=min(oversample, len(df)), random_state=config.get('seed', 42))

    # 5. Formiranje prompta
    df['prompt'] = "Generate a " + df['category'] + " quote:"
    
    # Pretvaranje u HuggingFace format
    formatted_data = [
        {
            "id": str(i),
            "translation": {src_lang: row['prompt'], tgt_lang: row['quote']}
        }
        for i, row in df.iterrows()
    ]

    dataset = HFDataset.from_list(formatted_data)
    return dataset


def load_data(
        source_language: str, 
        target_language: str
    ) -> HFDataset:
    """
    Translates the .tmx file into a HFDataset with given languages.

    Args:
        source_language (str): Original language of the dataset.
        target_language (str): Translated language of the dateset.

    Returns:
        HFDataset: HuggingFace dataset with columns id and translations.
                    id - the number of the current row.
                    translations - a dictionary with entries 'language': sentence,
                    which has at least two different languages.
    """
    # Open the tmx file.
    with open(f"{source_language}-{target_language}.tmx", "rb") as fin:
        tmx_file = tmxfile(fin, "en", "sr_Cyrl")

    # Define the data in the HuggingFace standard.
    data = {'id' : [], 'translation': []}
    i = 0

    # Iterate through the file and add the rows one by one.
    for item in tmx_file.unit_iter():

        data["id"].append(str(i))
        i = i + 1

        data["translation"].append({f"{source_language}": item.source.strip('"\n').lower(), f"{target_language}": item.target.strip('"\n').lower()})

    # Define the HuggingFace standard dataset.
    dataset = HFDataset.from_dict(data)

    return dataset
