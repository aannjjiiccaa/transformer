# Transformer Quote Generator

A custom **Transformer Encoder-Decoder** implementation in PyTorch designed to generate quotes based on specific styles or categories. By providing a "style" (e.g., *love, life, inspirational*) to the encoder, the model learns to generate contextually relevant and stylistically consistent quotes.

## Dataset
The model is trained on the [Quotes-500k Dataset from Kaggle](https://www.kaggle.com/datasets/manann/quotes-500k).

* **Columns:**
  * `quote`
  * `author`
  * `category` - multiple tags 
* **Data Processing:**
    * Trained on a randomized subset of **~100,000 entries**.
    * **Targeted Styles:** The model uses the first tag from the `category` column as the source input.
    * **Quality Control:** Extremely long or short quotes were excluded during preprocessing to maintain structural consistency and prevent padding-related noise.


## Features
* **Custom Transformer:** Built from scratch based on the "Attention Is All You Need" paper.
* **Optimized Architecture:** Tuned to 256 model dimensions to balance creativity and prevent memorization (overfitting).
* **Advanced Sampling:** Supports **Top-K** and **Temperature** sampling for diverse and natural text generation.
* **Automated Evaluation:** Integrated Perplexity, BLEU and METEOR metrics for performance tracking.
* **TensorBoard Integration:** Real-time monitoring of Training and Validation Loss and Evaluations.

## Architecture & Hyperparameters
Through rigorous testing, the following configuration was selected to optimize learning stability:

| Parameter | Value |
| :--- | :--- |
| **Model Dimension ($d_{model}$)** | 256 |
| **Heads ($h$)** | 8 |
| **Layers ($N$)** | 6 Encoder / 6 Decoder |
| **Context Size** | 96 tokens |
| **Dropout** | 0.3 |
| **Label Smoothing** | 0.15 |
| **Batch Size** | 128 - 256 |

## Training Analysis (Addressing Overfitting)
Initially, a larger model ($d_{model}=512$) showed significant overfitting after the 10th epoch, where Validation Loss began to diverge. 

**Solution:**
1. Reduced model capacity to **256 dimensions**.
2. Increased **Dropout** to 0.3.
3. Implemented a **Learning Rate Scheduler** (`ReduceLROnPlateau`) to handle the steep loss gradients seen in later epochs.


## Installation

### Kaggle Setup (GPU P100, Internet ON)
  1. Essential Environment Fixes
  ```bash
    !pip install -q "protobuf==3.20.3" --force-reinstall
    !pip install -q evaluate datasets
  ```
 2. Repo Management
  ```bash
  !git clone https://github.com/aannjjiiccaa/transformer.git
  %cd transformer
  ```

 3. Visualization & Training
  ```bash
  %load_ext tensorboard
  %tensorboard --logdir runs/quotes
  
  !python train.py
  ```

## Project Structure
- model.py: Core Transformer architecture (Attention, MultiHead, Encoder/Decoder).

- dataset.py: Data loading and custom Tokenizer logic.

- train.py: Training loop with validation and checkpointing.

- config.py: Centralized hyperparameters and path management.

- test.py: Inference engine with Top-K sampling logic.
