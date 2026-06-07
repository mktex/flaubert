import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset, DataLoader, RandomSampler


class TokenIdsDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, pos):
        assert pos < len(self.data) - self.block_size
        x = self.data[pos:(pos + self.block_size)]
        y = self.data[(pos + 1):(pos + 1 + self.block_size)]
        return x, y


class CharTokenizer:
    def __init__(self, vocabulary):
        self.token_id_for_char = {char: token_id for token_id, char in enumerate(vocabulary)}
        self.char_for_token_id = {token_id: char for token_id, char in enumerate(vocabulary)}
    @staticmethod
    def train_from_text(text):
        vocabulary = set(text)
        return CharTokenizer(sorted(list(vocabulary)))
    def encode(self, text):
        token_ids = []
        for char in text:
            token_ids.append(self.token_id_for_char[char])
        return torch.tensor(token_ids, dtype=torch.long)
    def decode(self, token_ids):
        chars = []
        for token_id in token_ids.tolist():
            chars.append(self.char_for_token_id[token_id])
        return "".join(chars)
    def vocabulary_size(self):
        return len(self.token_id_for_char)


class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], config["use_bias"])
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], config["use_bias"])
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], config["use_bias"])
        self.dropout = nn.Dropout(config["dropout_rate"])
        causal_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("causal_attention_mask", causal_attention_mask)
    def forward(self, input):
        batch_size, tokens_num, d_embed = input.shape
        Q = self.Q_weights(input)
        K = self.K_weights(input)
        V = self.V_weights(input)
        attention_scores = Q @ K.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            self.causal_attention_mask[:tokens_num, :tokens_num] == 0, -torch.inf
        )
        attention_scores = attention_scores / (K.shape[-1] ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)
        self.linear = nn.Linear(config["heads_num"] * config["d_embed"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])
    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]
        scores_change = torch.cat(heads_outputs, dim=-1)
        scores_change = self.linear(scores_change)
        return self.dropout(scores_change)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], config["d_embed"] * 4),
            nn.GELU(),
            nn.Linear(config["d_embed"] * 4, config["d_embed"]),
            nn.Dropout(config["dropout_rate"]),
        )
    def forward(self, input):
        return self.linear_layers(input)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])
    def forward(self, input):
        residual = input
        x = self.multi_head(self.layer_norm_1(input))
        x = x + residual
        residual = x
        x = self.feed_forward(self.layer_norm_2(x))
        return x + residual


class DemoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])
        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(config["d_embed"])
        self.unembedding = nn.Linear(config["d_embed"], config["vocabulary_size"], bias=False)
    def forward(self, token_ids):
        batch_size, tokens_num = token_ids.shape
        x = self.token_embedding_layer(token_ids)
        sequence = torch.arange(tokens_num, device=device)
        x = x + self.positional_embedding_layer(sequence)
        x = self.layers(x)
        x = self.layer_norm(x)
        x = self.unembedding(x)
        return x


def generate(model, prompt_ids, max_tokens):
    output_ids = prompt_ids
    for _ in range(max_tokens):
        if output_ids.shape[1] >= config["context_size"]:
            break
        with torch.no_grad():
            logits = model(output_ids)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # Sample a random token given the softmax distribution
        next_token_id = torch.multinomial(probs, num_samples=1)
        # Add new token to the output, and repeat the process
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)
    return output_ids


def generate_with_prompt(model, tokenizer, prompt, max_tokens=100):
    model.eval()
    prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to(device)
    return tokenizer.decode(generate(model, prompt, max_tokens=max_tokens)[0])


@torch.no_grad()
def calculate_validation_loss(model, batches_num):
    model.eval()
    total_loss = 0
    validation_iter = iter(test_dataloader)
    for _ in range(batches_num):
        idx, targets = next(validation_iter)
        logits = model(idx)
        logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
        targets_view = targets.view(batch_size * config["context_size"])
        loss = F.cross_entropy(logits_view, targets_view)
        total_loss += loss.item()
    average_loss = total_loss / batches_num
    return average_loss


# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[x] Device:", device)

# Beispiel Daten laden
with open("./data/tiny-shakespeare.txt", "rb") as f: text = str(f.read())

tokenizer = CharTokenizer.train_from_text(text)

# test Tokenizer
if True:
    test_text = "Hello World"
    print("[x] Input Text:", test_text)
    print("[x] => ", tokenizer.encode(test_text))
    print("[x] Die Umkehrung, Dasselbe wie:", tokenizer.decode(tokenizer.encode(test_text)))
    print(f"[x] Wörterbuchsgröße (Anzahl Symbole): {tokenizer.vocabulary_size()}")

# Test Dataset zusammenschustern
text_enc = tokenizer.encode(text)
trainset = TokenIdsDataset(text_enc, 32)
sampler = RandomSampler(trainset, replacement=True)
dataloader = DataLoader(trainset, batch_size=3, sampler=sampler)
xbatch, ytarget = next(iter(dataloader))
[tokenizer.decode(x) for x in xbatch]
[tokenizer.decode(x) for x in ytarget]

# 256, 768, 12, 10, 0.1, False
config = {
    "vocabulary_size": tokenizer.vocabulary_size(),
    "context_size": 64,
    "d_embed": 15,
    "heads_num": 3,
    "layers_num": 5,
    "dropout_rate": 0.1,
    "use_bias": False,
}
config["head_size"] = config["d_embed"] // config["heads_num"]

# Test Komponenten
if True:
    input = torch.rand(8, config["context_size"], config["d_embed"])
    ah = AttentionHead(config)
    mha = MultiHeadAttention(config)
    ff = FeedForward(config)
    b = Block(config)
    output_ah = ah(input)
    output_mha = mha(input)
    output_ff = ff(input)
    output_block = b(input)
    print("[x] Attention Head:", output_ah.shape)
    print("[x] Multi-Head Attention:", output_mha.shape)
    print("[x] Feed-Forward:", output_ff.shape)
    print("[x] Block:", output_block.shape)

# teste die rechennetzwerk
if True:
    output = output_block
    logits = output[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    print(next_token_id)

# Modellierung
batch_size = 8
train_iterations = 100
evaluation_interval = 100
learning_rate = 4e-4
train_split = 0.9
model = DemoGPT(config).to(device)

# Test Model-Output
if True:
    output = model(tokenizer.encode("Hi").unsqueeze(dim=0).to(device))
    generate_with_prompt(model, tokenizer, "First Citizen:")

# Setup Train-Test Data
if True:
    tokenized_text = tokenizer.encode(text).to(device)
    ntokens_training = int(train_split * tokenized_text.shape[0])
    train_data = tokenized_text[:ntokens_training]
    test_data = tokenized_text[ntokens_training:]
    train_dataset = TokenIdsDataset(train_data, config["context_size"])
    validation_dataset = TokenIdsDataset(test_data, config["context_size"])
    train_sampler = RandomSampler(train_dataset, num_samples=batch_size * train_iterations, replacement=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_sampler = RandomSampler(validation_dataset, replacement=True)
    test_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

# Teste Validierung
if True:
    validation_iter = iter(test_dataloader)
    idx, targets = next(validation_iter)
    calculate_validation_loss(model, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Prozedur
"""
import os
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from IPython.display import display
import ipywidgets as widgets
%matplotlib inline
plot_output = widgets.Output()
display(plot_output)

def update_plot(train_losses, train_steps, validation_losses, validation_steps):
  with plot_output:
    clear_output(wait=True)  # Clear only the plot output, not the text
    plt.figure(figsize=(7, 5))
    plt.plot(train_steps, train_losses, label='Training Loss')
    plt.plot(validation_steps, validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.legend(loc='center left')
    plt.grid(True)
    plt.show()
"""

# Teste Trainingprozedur für 10 Schritte
train_losses = []
train_steps = []
eval_losses = []
eval_steps = []
if True:
    for epoch in range(100):
        for step_num, sample in enumerate(train_dataloader):
            _ = model.train()
            inputs, targets = sample
            logits = model(inputs)
            logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
            targets_view = targets.view(batch_size * config["context_size"])
            loss = F.cross_entropy(logits_view, targets_view)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_steps.append(step_num)
            if step_num % evaluation_interval == 0:
                print("\nBeispiel:\n" + generate_with_prompt(model, tokenizer, "So he "))
                print(f"Step {step_num}. Loss {loss.item():.3f}")
                validation_loss = calculate_validation_loss(model, batches_num=10)
                eval_losses.append(validation_loss)
                eval_steps.append(step_num)
                print(f"Step {step_num}. Validation loss: {validation_loss:.3f}")
            # if step_num > 10: break
            # update_plot(train_losses, train_steps, eval_losses, eval_steps)












