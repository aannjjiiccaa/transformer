# Torch stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Other files stuff
from dataset import BilingualDataset, load_data_quotes
from model import get_model
from config import get_weights_file_path, get_latest_weights, get_config
from test import load_model_and_tokenizers, run_validation, run_validation_teacher_forcing, run_validation_visualization, run_test, generate_quote

# HuggingFace stuff
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit, Punctuation
from datasets import Dataset as HFDataset

# Metrics stuff
import warnings

# Easy access stuff
from pathlib import Path
from tqdm import tqdm

# Set the random seed for this project, for reproducibility.
import random
SEED = get_config()["seed"]
torch.manual_seed(SEED)
random.seed(SEED)

            
def get_all_sentences(
        dataset: HFDataset,
        language: str
    ):
    """
    Yields elements of the provided dataset.

    Args:
        dataset (HFDataset): Dataset to iterate through.
        language (str): Language given as a language code, present in the dataset.

    Yields:
        str: Sentence from the dataset in the provided language.
    """
    for item in dataset:
        yield item['translation'][language]
        

def get_or_build_tokenizer(
        config, 
        dataset: HFDataset, 
        language: str,
        force_rewrite: bool = False,
        min_frequency: int = 5,
        vocab_size: int = 1000000
    ) -> Tokenizer:
    """ 
    If the path to tokenizer file is not specified in the config, or if
    we force rewrite, then build a tokenizer from scratch.
    Else, get the tokenizer from the specified file.

    Args:
        config: A config file.
        dataset (HFDataset): HuggingFace dataset of translations to build the tokenizer from.
        language (str): Language from the dataset for the tokenizer.
        force_rewrite (bool): If the function should disregard the config file.
        min_frequency (int): Minimum frequency of a word in the dataset to add it to the vocabulary.
        vocab_size (int): Maximum size of the vocabulary.

    Returns:
        Tokenizer: A tokenizer for the specified language built from a vocabulary formed by sentences from the dataset.
    """
    # Get the path from config.
    tokenizer_path = Path(config['tokenizer_file'].format(language))

    # If such a path doesn't exist, or we force the rewrite, then build the tokenizer.
    if not Path.exists(tokenizer_path) or force_rewrite:

        # Initialize the tokenizer with the unknown token [UNK].
        # This tokenizer will consider full words to be tokens.
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

        # Build a trainer with the specified special tokens, and parameters for minimum frequency and vocabulary size.
        trainer = WordLevelTrainer(
            special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency = min_frequency, 
            vocab_size = vocab_size
            )
        
        # Train the tokenizer on the dataset in the given language.
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer = trainer)

        # Save the tokenizer to file.
        tokenizer.save(str(tokenizer_path))

    # Get the tokenizer from file.
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Return the tokenizer and print the number of tokens in it.
    print(f"Number of tokens in {language} is {tokenizer.get_vocab_size()}.")
    return tokenizer


def get_dataset(config):
    """
    Initializes the training and validation datasets.
    Initializes the tokenizers in each language.

    Args:
        config: A config file.

    Returns:
        DataLoader: Training dataset dataloader.
        DataLoader: Validation dataset dataloader.
        Tokenizer: Source language tokenizer.
        Tokenizer: Target language tokenizer.
    """
    # Load the data from the csv file.
    dataset_raw = load_data_quotes(
        csv_file="/kaggle/input/datasets/manann/quotes-500k/quotes.csv",
        src_lang=config['source_language'],
        tgt_lang=config['target_language'],
        config=config,
        sample_size=config.get('sample_size', 30000)
    )

    # Initialize the training, validation and test dataset sizes.
    dataset_size = len(dataset_raw)
    train_dataset_size = int(0.9 * dataset_size)
    validation_dataset_size = int(0.09 * dataset_size)
    test_dataset_size = dataset_size - train_dataset_size - validation_dataset_size
    train_raw, val_raw, test_raw = random_split(
        dataset_raw, [train_dataset_size, validation_dataset_size, test_dataset_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    # Split the data into datasets.
  #  training_dataset_raw, validation_dataset_raw, test_dataset_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size, test_dataset_size])

    source_tokenizer = get_or_build_tokenizer(config, train_raw, config['source_language'])
    target_tokenizer = get_or_build_tokenizer(config, train_raw, config['target_language'])

    # Define the BilingualDataset objects for the training and validation datasets.
    training_dataset = BilingualDataset(train_raw, source_tokenizer, target_tokenizer, config['source_language'], config['target_language'], config['context_size'])
    validation_dataset = BilingualDataset(val_raw, source_tokenizer, target_tokenizer, config['source_language'], config['target_language'], config['context_size'])
    test_dataset = BilingualDataset(test_raw, source_tokenizer, target_tokenizer, config['source_language'], config['target_language'], config['context_size'])

    # Calculate the maximum lengths of the sentences in training dataset.
    # Only for testing purposes.
    # max_len_source = 0
    # max_len_target = 0
    # for item in dataset_raw:
    #     source_ids = source_tokenizer.encode(item['translation'][config['source_language']]).ids
    #     target_ids = target_tokenizer.encode(item['translation'][config['target_language']]).ids
    #     max_len_source = max(max_len_source, len(source_ids))
    #     max_len_target = max(max_len_target, len(target_ids))

    # print(f"Max length of source sentence: {max_len_source}")
    # print(f"Max length of target sentence: {max_len_target}")

    # Define the DataLoader objects for training and validation datasets.
    training_dataloader = DataLoader(training_dataset, batch_size = config['batch_size'], shuffle = True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = config['batch_size'], shuffle = False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    return training_dataloader, validation_dataloader, test_dataloader, source_tokenizer, target_tokenizer


def train_model(config):
    """
    Train the transformer model with the given parameters.

    Args:
        config: A config file.
    """
    # Use cuda if possible, otherwise use cpu.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}.')

    # Make the folder for the model weights.
    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    # Get the datasets and define the model.
    training_dataloader, validation_dataloader, test_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)
    print(model)
    # Initialize the writer to visualize data.
    writer = SummaryWriter(config['experiment_name'])

    # Initialize the Adam optimizer.
    # Adjusts the learning rate as the model trains.
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'], eps = 1e-9,weight_decay=1e-5)
    scaler = torch.GradScaler()
    # Load a pretrained model if defined and if it exists.
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = get_latest_weights(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from the beginning.")

    # Define the loss function.
    loss_function = nn.CrossEntropyLoss(ignore_index = target_tokenizer.token_to_id('[PAD]'), label_smoothing = 0.15).to(device)
    best_loss = float('inf')
    best_model_path = Path(config['model_folder']) / "best_model.pt"
    if best_model_path.exists():
        state = torch.load(best_model_path)
        best_loss = state.get('val_loss', float('inf'))

    # Run the epochs.
    for epoch in range(initial_epoch, config['num_epochs']):
        
        # Create a batch iterator and iterate through the batches.
        batch_iterator = tqdm(training_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            # Put the model in the training state.
            model.train()

            # Move the tensors to the device.
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                # Calculate the outputs of the model for this batch.
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                transformer_output = model.project(decoder_output)
                loss = loss_function(transformer_output.view(-1,target_tokenizer.get_vocab_size()),label.view(-1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_iterator.set_postfix({"loss":f"{loss.item():6.3f}"})
            writer.add_scalar('train_loss',loss.item(),global_step)
            # Adjust the global step.
            global_step += 1

        # Run the validation at the end of every epoch.
        run_validation_visualization(model, validation_dataloader, source_tokenizer, target_tokenizer, config['context_size'], device, lambda msg: batch_iterator.write(msg), writer, global_step, number_examples = 1)
        # run_validation_teacher_forcing(model, validation_dataloader, loss_function, target_tokenizer, device)
        avg_val_loss=run_validation(model, validation_dataloader, loss_function, target_tokenizer, device, epoch, source_tokenizer)

        # Save weights at certain 'milestone' epochs.
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'val_loss': avg_val_loss
            }, best_model_path)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_dataloader, validation_dataloader, test_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)
    best_model_path = Path(config['model_folder']) / "best_model.pt"
    if best_model_path.exists():
        print(f"Loading best weights from {best_model_path} for final testing...")
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    
    model.eval()
    print("\n--- FINAL EVALUATION ON BEST MODEL ---")
    loss_function = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id('[PAD]')).to(device)
    run_validation(model, validation_dataloader, loss_function, tgt_tokenizer, device, "FINAL", src_tokenizer)
    run_test(model, test_dataloader, src_tokenizer, tgt_tokenizer, device)
    
    # Vizuelizacija 10-20 primera (50 je možda previše za čitanje odjednom)
    run_validation_visualization(model, validation_dataloader, src_tokenizer, tgt_tokenizer, config['context_size'], device, print, None, 0, number_examples=10)
