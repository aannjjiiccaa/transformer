from pathlib import Path
Path("weights").mkdir(exist_ok=True)


def get_config():
    """
    Returns:
        A static dictionary of model configuration variables:
            batch_size (int): batch size of the model
            num_epochs (int): number of epochs of the model
            learning_rate (float): learning rate of the model
            context_size (int): maximum allowed sentence length (in tokens)
            model_dimension (int): dimension of the embedding vector space
            source_language (str): original language of the sentences, as a language code
            target_language (str): translated language of the sentences, as a language code
            model_folder (str): folder in which the weights will be saved
            model_basename (str): name of the model
            preload (int | None): epoch from which to load the weights
            tokenizer_file: files where the tokenizers are stored
            experiment_name: tensorboard experiment name
            seed: (int | None): seed of the model
    """
    return {
        "batch_size": 128,
        "num_epochs": 100,
        "learning_rate": 1 * 10**-4,
        "context_size": 96,
        "model_dimension": 256,
        "source_language": "src",
        "target_language": "tgt",
        "model_folder": "weights",
        "model_basename": "quote_gen",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/quotes",
        "seed": 561,
        "sample_size":200000
    }

def get_weights_file_path(
        config, 
        epoch: str
    ) -> str:
    """
    Get the saved model weights from a file.

    Args:
        config: Config file.
        epoch (str): Epoch from which to load the weights.

    Returns:
        str: Path to the saved weights of the model.
    """

    # Find the appropriate file.
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)

def get_latest_weights(config) -> str:
    """
    Get the latest saved model weights from a folder.

    Args:
        config: Config file.

    Returns:
        str: Path to the latest saved weights of the model.
    """

    # Find all the files in the folder.
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}*"
    model_filenames = list(Path(model_folder).glob(model_filename))

    # If the folder is empty then there is nothing to return.
    if len(model_filenames) == 0:
        return None
    
    # Define a key for sorting. Extracts the epoch int from the filename.
    def extract_epoch(filename):
        return int(filename.stem.split('_')[-1])
    
    # Sort the files by their epoch.
    model_filenames.sort(key = extract_epoch)

    return str(model_filenames[-1])
