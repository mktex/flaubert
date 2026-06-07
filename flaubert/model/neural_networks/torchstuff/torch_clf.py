import os
import pandas as pd

# Define paths to dataset
train_pos_path = './aclImdb/train/pos' # Path to the directory containing positive reviews from the training set
train_neg_path = './aclImdb/train/neg' # Path to the directory containing negative reviews from the training set
test_pos_path = './aclImdb/test/pos' # Path to the directory containing positive reviews from the test set
test_neg_path = './aclImdb/test/neg' # Path to the directory containing negative reviews from the test set

import os
# os.system("ls ./aclImdb/train/pos")
with open(f'{train_pos_path}/1008_10.txt', 'r') as f:
    one_file_content = f.read()

print(one_file_content)

import os


def for_schleife_mit_else():
    """ Beispiel """
    for i in range(5):
        print(i)
    else:
        print("Loop just finished!")


def do_generator():
    """ Beispiel """
    def iclock():
        ik = 1
        while ik <= 4:
            yield ik
            ik += 1
            if ik == 5: ik = 1

    xc = iclock()
    next(xc)


def xtract_content(subdir, xfile):
    with open(f'{subdir}/{xfile}'.replace('//', '/'), 'r') as f:
        one_file_content = f.read()
    return one_file_content


def load_dataset(xfiles_path):
    """
    Reads all text files in the specified folder and returns their content as a list.
    Args:
        folder (str): Path to the folder containing text files.
    Returns:
        list: A list of strings, where each string is the content of a text file.
    """
    subdir, dirs, files = list(os.walk(xfiles_path))[0]
    txt_files = list(filter(lambda x: '.txt'==x[-4:], files))
    print(f"[x] Extracting from {xfiles_path} {len(txt_files)} files..")
    return list(map(lambda fpath: xtract_content(subdir, fpath), txt_files))


train_pos = load_dataset(train_pos_path)
print(train_pos[42])
train_neg = load_dataset(train_neg_path)
print(train_neg[42])
test_pos = load_dataset(test_pos_path)
print(test_pos[42])
test_neg = load_dataset(test_neg_path)
print(test_neg[42])


# Create DataFrames
train_df = pd.DataFrame({
    'review': train_pos + train_neg,
    'label': [1] * len(train_pos) + [0] * len(train_neg)
})

test_df = pd.DataFrame({
    'review': test_pos + test_neg,
    'label': [1] * len(test_pos) + [0] * len(test_neg)
})

print(train_df.sample(5))

# Assert that both datasets have the expected number of rows
assert train_df.shape[0] == 25000, "Training dataset does not have 25000 rows."
assert test_df.shape[0] == 25000, "Testing dataset does not have 25000 rows."

# Assert that both datasets have exactly two columns
assert train_df.shape[1] == 2, "Training dataset does not have exactly 2 columns."
assert test_df.shape[1] == 2, "Testing dataset does not have exactly 2 columns."


def get_simple_stats(xreview):
    nchars = len(list(filter(lambda x: x != " ", xreview)))
    nwords = len(xreview.split(" "))
    return nchars, nwords

train_simple_stats_list = list(map(lambda xreview: get_simple_stats(xreview),
                                   train_df['review'].values))

test_simple_stats_list = list(map(lambda xreview: get_simple_stats(xreview),
                                   test_df['review'].values))

train_df["nchars"] = [w[0] for w in train_simple_stats_list]
train_df["nwords"] = [w[1] for w in train_simple_stats_list]

test_df["nchars"] = [w[0] for w in test_simple_stats_list]
test_df["nwords"] = [w[1] for w in test_simple_stats_list]

train_df.info()
train_df.describe()

import matplotlib.pyplot as plt

t = train_df.groupby('label').count()
t.plot(kind="bar", color="cyan")
plt.xticks(rotation=0)
plt.tight_layout()
plt.title("Train Counts")


test_df.info()
test_df.describe()

t = test_df.groupby('label').count()
t.plot(kind="bar", color="cyan")
plt.xticks(rotation=0)
plt.tight_layout()
plt.title("Test Counts")

# Number of characters
train_df["nchars"].hist(bins=45, color="black", label="train")
test_df["nchars"].hist(bins=45, color="gray", alpha=0.5, label="test")
plt.title("Histograms of length of chars")
plt.legend()
plt.tight_layout()

# Number of words
train_df["nwords"].hist(bins=45, color="black", label="train")
test_df["nwords"].hist(bins=45, color="gray", alpha=0.5, label="test")
plt.title("Histograms of length of words")
plt.legend()
plt.tight_layout()

# positive reviews
record = train_df[train_df['label']==1].sample(10).iloc[5]
print(record)
print(record.review)

# negative reviews
record = train_df[train_df['label']==0].sample(10).iloc[5]
print(record)
print(record.review)

# Split train data into training and validation sets manually
train_size = int(0.9 * len(train_df))

# Shuffle the dataset
shuffled_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_data = shuffled_df.iloc[:train_size]
val_data = shuffled_df.iloc[train_size:]

from transformers import AutoTokenizer
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer

# Take sample inputs from the dataset
sample_texts = train_data['review'].sample(3, random_state=42).tolist()

# Tokenize sample inputs
tokenized_samples = tokenizer(sample_texts, truncation=True,
                              padding="max_length", max_length=128,
                              return_tensors="pt")
tokenizer(train_data['review'].iloc[42], truncation=True,
          padding="max_length", max_length=128,
          return_tensors="pt")['input_ids'].shape

tokenizer(train_data['review'].iloc[42], truncation=True,
          padding="max_length", max_length=128,
          return_tensors="pt")['input_ids'][0, :].shape

import torch
from torch.utils.data import Dataset
MAX_LENGTH = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    """
    A custom PyTorch Dataset for the IMDB dataset.
    This class preprocesses text data using a tokenizer and returns tokenized inputs
    along with their corresponding labels for sentiment analysis.
    Attributes:
        data (pd.DataFrame): A DataFrame containing text and label columns.
        tokenizer (transformers.PreTrainedTokenizer):
            The tokenizer used for preprocessing text.
        max_length (int): Maximum length for tokenized sequences.
    """
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        """
        Initialize the dataset.
        Args:
            data (pd.DataFrame): A DataFrame with columns `review` (text)
                and `label` (target).
            tokenizer (transformers.PreTrainedTokenizer):
                The tokenizer to preprocess the text.
            max_length (int, optional): Maximum token sequence length. Defaults to 128.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a single data point by index and preprocess it.
        Args:
            idx (int): Index of the data point to retrieve.
        Returns:
            torch.Tensor: Tokenized input IDs for the text.
            int: Label corresponding to the text.
        """
        input_text = self.data['review'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return self.tokenizer(input_text,
                              truncation=True, padding="max_length",
                              max_length=self.max_length,
                              return_tensors="pt")['input_ids'][0, :], label


# Initialize the datasets
train_dataset = IMDBDataset(train_data, tokenizer)
train_dataset

val_dataset = IMDBDataset(val_data, tokenizer)
val_dataset

test_dataset = IMDBDataset(test_df, tokenizer)
test_dataset

from torch.utils.data import DataLoader

# Define batch size
BATCH_SIZE = 32

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader

# check iteration
xbatch, ytarget = next(iter(train_loader))

print(xbatch.shape, ytarget.shape)
xbatch[0], ytarget[0]

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader

assert len(train_dataset) == 22500, "Train dataset length mismatch!"
assert len(val_dataset) == 2500, "Validation dataset length mismatch!"
assert len(test_dataset) == 25000, "Test dataset length mismatch!"

import numpy as np

# Check the first item in the train dataset
input_ids, label = train_dataset[0]
assert isinstance(input_ids, torch.Tensor), "Input IDs should be a torch.Tensor!"
assert isinstance(label, (int, np.integer)), "Label should be an integer or int-like!"

# Ensure the input IDs tensor has the correct shape
assert input_ids.shape[0] == train_dataset.max_length, "Input IDs tensor has incorrect length!"

from pprint import pprint
config = {
    "vocabulary_size": tokenizer.vocab_size,  # e.g., ~30522 for bert-base-uncased
    "num_classes": 2,                         # binary classification (pos/neg)
    "d_embed": 128,
    "context_size": MAX_LENGTH,
    "layers_num": 4,
    "heads_num": 4,
    "head_size": 32,  # 4 heads * 32 = 128 -> matches d_embed
    "dropout_rate": 0.1,
    "use_bias": True
}

pprint(config)

import torch.nn as nn
import math


class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        causal_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("causal_attention_mask", causal_attention_mask)

    def forward(self, input):
        batch_size, tokens_num, d_embed = input.shape
        Q = self.Q_weights(input)  # (B, T, head_size)
        K = self.K_weights(input)  # (B, T, head_size)
        V = self.V_weights(input)  # (B, T, head_size)

        # Q @ K^T => (B, T, T)
        attention_scores = Q @ K.transpose(1, 2)

        # Causal Mask
        attention_scores = attention_scores.masked_fill(
            self.causal_attention_mask[:tokens_num, :tokens_num] == 0,
            -torch.inf
        )
        attention_scores = attention_scores / math.sqrt(K.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V


# Instantiate the AttentionHead
attention_head = AttentionHead(config).to(device)

# Create a dummy input of shape (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE,
                          config["context_size"],
                          config["d_embed"]).to(device)

# Forward pass
attention_output = attention_head(dummy_input)
print("AttentionHead output shape:", attention_output.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)
        self.linear = nn.Linear(config["heads_num"] * config["head_size"],
                                config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]
        x = torch.cat(heads_outputs, dim=-1)  # (B, T, heads_num * head_size)
        x = self.linear(x)                   # (B, T, d_embed)
        x = self.dropout(x)
        return x

# Instantiate MultiHeadAttention
multi_head_attention = MultiHeadAttention(config).to(device)

# Same dummy input: (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE, config["context_size"],
                          config["d_embed"]).to(device)

# Forward pass
mha_output = multi_head_attention(dummy_input)
print("MultiHeadAttention output shape:", mha_output.shape)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input):
        return self.linear_layers(input)

# Instantiate FeedForward
feed_forward = FeedForward(config).to(device)

# Dummy input: (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE, config["context_size"],
                          config["d_embed"]).to(device)

# Forward pass
ff_output = feed_forward(dummy_input)
print("FeedForward output shape:", ff_output.shape)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

    def forward(self, input):
        x = input
        x = x + self.multi_head(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

# Instantiate a single Block
block = Block(config).to(device)

# Dummy input: (32, 128, 128)
dummy_input = torch.randn(BATCH_SIZE, config["context_size"],
                          config["d_embed"]).to(device)

# Forward pass
block_output = block(dummy_input)
print("Block output shape:", block_output.shape)


class DemoGPT(nn.Module):
    def __init__(self, config):
        """
        Initialize the DemoGPT class with configuration parameters.

        Args:
        - config (dict): Configuration dictionary with the following keys:
            - "vocabulary_size": Size of the vocabulary.
            - "d_embed": Dimensionality of the embedding vectors.
            - "context_size": Maximum sequence length (context size).
            - "layers_num": Number of transformer layers.
            - "num_classes": Number of output classes (2 for binary classification).
        """
        super().__init__()
        # Token embedding layer: Maps token indices to embedding vectors.
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"],
                                                  config["d_embed"])

        # Positional embedding layer: Adds positional information to the embeddings.
        self.positional_embedding_layer = nn.Embedding(config["context_size"],
                                                       config["d_embed"])

        # Transformer layers: Stacked sequence of transformer blocks.
        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)

        # Layer normalization: Applied to stabilize training.
        self.layer_norm = nn.LayerNorm(config["d_embed"])

        # TODO: Implement classification output layer - Maps pooled embeddings to class logits.
        self.unembedding = nn.Linear(config["d_embed"],
                                     config["num_classes"], bias=False)

    def forward(self, token_ids):
        """
        Forward pass of the model.
        Args:
        - token_ids (torch.Tensor): Input token indices of shape (B, T),
                                    where B is the batch size,
                                    and T is the sequence length.

        Returns:
        - logits (torch.Tensor): Output logits of shape (B, num_classes).
        """
        batch_size, tokens_num = token_ids.shape

        # Step 1: Create embeddings for tokens and their positions
        x = self.token_embedding_layer(token_ids)  # Shape: (B, T, d_embed)
        positions = torch.arange(tokens_num, device=token_ids.device)  # Shape: (T,)
        pos_embed = self.positional_embedding_layer(positions)  # Shape: (T, d_embed)
        x = x + pos_embed.unsqueeze(0)  # Add positional embeddings to token embeddings

        # Step 2: Pass embeddings through transformer layers
        x = self.layers(x)  # Shape: (B, T, d_embed)
        x = self.layer_norm(x)  # Normalize across the feature dimension

        # Step 3: TODO: Apply mean pooling across the time dimension  # Shape: (B, d_embed)
        x = self.unembedding(x)

        # Step 4: TODO: Generate logits for classification  # Shape: (B, num_classes)
        logits = torch.mean(x, dim=1)

        return logits


# Instantiate the model
demo_gpt = DemoGPT(config).to(device)

# Suppose we have a batch of size 32, each with a sequence length of 128
dummy_token_ids = torch.randint(
    0, config["vocabulary_size"],
    (BATCH_SIZE, config["context_size"])
).to(device)

# Forward pass
logits = demo_gpt(dummy_token_ids)

print("DemoGPT output shape:", logits.shape)
print("Logits sample:\n", logits[:2])  # Print first two examples' logits

# Assert that the number of logits matches the number of classes
assert logits.size(1) == config["num_classes"], (
    f"Expected number of classes {config['num_classes']}, "
    f"but got {logits.size(1)}"
)

# Assert that the batch size of the output matches the input batch size
assert logits.size(0) == BATCH_SIZE, (
    f"Expected batch size {BATCH_SIZE}, "
    f"but got {logits.size(0)}"
)
import torch.nn.functional as F

def calculate_accuracy(model, data_loader, device):
    """
    Calculate the accuracy of the model on the validation dataset.
    Args:
        model (torch.nn.Module): The trained transformer model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model (e.g., 'cuda' or 'cpu').
    Returns:
        float: Validation accuracy as a percentage.
    """
    model.eval()
    validation_iter = iter(val_loader)
    bootstrap_means = []
    for batch, targets in iter(validation_iter):
        with torch.no_grad():
            logits = model(batch.to(device))
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        acc = np.mean(preds.to("cpu").numpy() == targets.to("cpu").numpy()) * 100
        bootstrap_means.append(acc)
    accuracy = np.mean(bootstrap_means)
    return accuracy


model = DemoGPT(config).to(device)

validation_accuracy = calculate_accuracy(model, val_loader, device)
print(f"Validation Accuracy: {validation_accuracy:.2f}%")

import torch.optim as optim

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
EPOCHS = 5

# Initialize model, loss, and optimizer
model = DemoGPT(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(EPOCHS):

    print("=" * 80)
    print(f"[x] Epoch {epoch}")

    model.train()
    running_loss = 0.0

    for step, (input_ids, labels) in enumerate(train_loader):

        batch_size = input_ids.shape[0]
        if batch_size != BATCH_SIZE:
            print(f"[x] Skipped uncomplete batch ({batch_size})")
            continue

        # Move data to device
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # TODO: Implement forward pass
        logits = model(input_ids)
        logits_view = logits.view(batch_size, config["num_classes"])

        # TODO: Calculate loss
        targets_view = labels.view(batch_size)
        loss = F.cross_entropy(logits_view, targets_view)

        # TODO: Set gradients to zero
        optimizer.zero_grad(set_to_none=True)

        # TODO: Backward pass
        loss.backward()

        # TODO: Step the optimizer
        optimizer.step()

        running_loss += loss.item()

        # Log training progress
        if (step + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{step + 1}/{len(train_loader)}], "
                  f"Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # Evaluate validation accuracy
    val_accuracy = calculate_accuracy(model, val_loader, device)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {val_accuracy:.2f}%")


# TODO: Calculate the accuracy of the model over the test set using the calculate_accuracy() function
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")

"""
# In Transformer die EmbeddingLayer hat auch einen Positional Encoding Layer
# EmbeddingLayer produziert die gleiche Resultate für selber Input mit Positional Encoding Layer 
# werden die Beziehungen gelernt: also "abc" nicht Dasselbe wie "bac"
	Q = Wq * Embed
	K = Wk * Embed
	V = Wv * Embed
Attention Scores = Q * K(transponiert)
Attention Scores Matrize muss so: AS[1, 1] = Q1 * K1(transponiert).
Durch Multiplikation der Matrizen Q, K sind die Qi auf Spalten und Kj auf Zeilen.
=> Softmax(Attention Scores) per Columns, so dass per Column SUM(.) = 1
V wird mit AS multipliziert aber als V1 * AS[i, 1], V2 * AS[i, 2], .. resultiert 
Die AS Matrize fungiert als Gewichtung auf Werte aus Vi. 
=> heißt die Summe SUM(Vi) wird mit Werte aus Spalte AS[i, :] gewichtet.
Der Resultat ist eine Aktualisierung der Embeddings
Die Matrize AS wird trianguliert, so dass die untere Hälfte auf Null gesetzt wird 
=> Beim Ausmultiplizieren für die Gewichtung der Vektoren Vi zählen nur die "vergangene" Positionen, 
     als V1 wird nur mit AS[1,1] gewichtet weil: V1 * AS[1, 1] + V2 * AS[2, 1] + V3 * AS[3, 1] .. 
     aber AS[j, 1] ist 0 für alle j > 1. 
     Der Wert wird -unendlich, damit die Softmax Funktion die 0 rausgibt.

Leseliste:
	Attention is all you need, Vaswani et al 2023, https://arxiv.org/abs/1706.03762
	PointWise NN, Shoeff et all 2019: https://arxiv.org/abs/1901.04544

"""





