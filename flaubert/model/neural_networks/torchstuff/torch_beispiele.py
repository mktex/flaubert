import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, RandomSampler

from transformers import AutoTokenizer
from transformers import pipeline

import pandas as pd
import numpy as np
from collections import OrderedDict

def softmax(z):
    return torch.exp(z) * torch.div(1, torch.sum(torch.exp(z), axis=1)).reshape(-1, 1)


def activation(x):
    return 1 / (1 + torch.exp(-x))


class Network1(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x


def get_accuracy_batch1(images, labels):
    ps = torch.exp(model(images))
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    return accuracy.item()


def get_accuracy1():
    acc_list = []
    for test_images, test_labels in iter(testloader):
        acc_list.append(get_accuracy_batch1(test_images, test_labels))
    res_accuracy = np.mean(acc_list)
    return res_accuracy


def get_accuracy_batch2(xmodel, _images, _labels):
    ps = torch.exp(xmodel(imgfunk(_images)))
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == _labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    return accuracy.item()


def get_accuracy2(xmodel, xloader):
    with torch.no_grad():
        xmodel.eval()
        acc_list = []
        for _images, _labels in iter(xloader):
            acc_list.append(
                get_accuracy_batch2(xmodel, _images, _labels)
            )
        res_accuracy = np.mean(acc_list)
        return res_accuracy


def classifier():
    _model = nn.Sequential(nn.Linear(784, 128),
                           nn.ReLU(),
                           nn.Linear(128, 64),
                           nn.ReLU(),
                           nn.Linear(64, 10),
                           nn.Dropout(p=0.2),
                           nn.LogSoftmax(dim=1))
    return _model


def apply_classifier(xmodel, _images):
    return torch.exp(model(imgfunk(images)))


def check_model():
    # Beispiel Training mit Transferlearning
    global model, testloader, running_loss
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / print_every:.3f}.. "
              f"Test loss: {test_loss / len(testloader):.3f}.. "
              f"Test accuracy: {accuracy / len(testloader):.3f}")
    model.train()


# Beispiel Char-Tokenizer
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
        return ''.join(chars)

    def vocabulary_size(self):
        return len(self.token_id_for_char)


class TokenIdsDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, pos):
        assert pos < len(self.data) - self.block_size - 1
        x = self.data[pos:(pos + self.block_size)]
        y = self.data[(pos + 1):(pos + 1 + self.block_size)]
        return x, y


# Multihead Attention Implementierung
config = {
    "vocabulary_size": tokenizer.vocabulary_size(),
    "context_size": 256,
    "embedding_dim": 768,
    "heads_num": 12,
    "layers_num": 10,
    "dropout_rate": 0.1,
    "use_bias": False,
}
config["head_size"] = config["embedding_dim"] // config["heads_num"]


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
        self.linear = nn.Linear(config["d_embed"], config["d_embed"])
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


imgfunk = lambda images: images

# einfache Netz in pytorch
torch.manual_seed(7)

# 3 Features zufällig
features = torch.randn((1, 3))

# torch basics
torch.rand(5)
torch.rand(5, 5)  # 5x5 matrize
torch.ones(5)
torch.ones(5, 5)
torch.arange(10)
torch.tril(torch.rand(5, 5))  # triangular matrix, lower != 0 rest 0
torch.cat((torch.rand(5, 5), torch.rand(5, 5)), axis=0)  # concatenate on rows
x = torch.arange(10)
x_star = x.view(5, 2)
x_star.view(10)

# Layers
n_input = features.shape[1]
n_hidden = 2
n_output = 1

# Gewichtungen und Biases
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# Aktivierung
hout = activation(torch.mm(features, W1) + B1)
out = activation(torch.mm(hout, W2) + B2)

# build it
model = Network1()


# *** Daten laden ***
transform1 = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
transform2 = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
# Beispiel Transformer mit Augmentation
train_transforms3 = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])
                                        ]
                                       )
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform1)

# Daten mit einem Transformer laden

dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Schleifen
images, labels = next(iter(dataloader))
for images, labels in dataloader: pass

# Gegeben einen Input Folder mit Bildern
# jede Klasse in eigenem Ordner (cats and dogs): ./dog/xyz.png or ./cat/abc.png
data_dir = 'Cat_Dog_data/train'
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
images, labels = next(iter(dataloader))

# Beispiel voll
data_dir = 'Cat_Dog_data'
train_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.RandomRotation(30),
                                       transforms.CenterCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()
                                       ])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()
                                      ])
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
data_iter = iter(trainloader)
images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    plt.imshow(images[ii], ax=ax, normalize=False)


# Beispiel Anwendung mit Sequential:
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
nsize, dim0, dim1 = 64, 1, 784
images_resized = images.resize_(nsize, dim0, dim1)
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0, :])
plt.imshow(images[42].numpy().squeeze(), cmap='Greys_r')

# Die Liste der verschiedenen Loss-Funktionen aus Torch:
# https://docs.pytorch.org/docs/stable/nn.html

# Einfacher Beispiel mit Cost-Funktion
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
print(loss)
# Zugriff auf Gradient:
print(model[0].weight.grad)

# Training Prozedur:
print('Initial weights - ', model[0].weight)
images, labels = next(iter(trainloader))
images.resize_(64, 784)
optimizer.zero_grad()
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', model[0].weight.grad)
optimizer.step()
print('Updated weights - ', model[0].weight)

# Konvertierung zwischen numpy und torch
a = torch.ones(5)
b = a.numpy()
a = np.array([1, 2, 3])
t = torch.from_numpy(a)

# Simple Train Prozedur
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# Alternative: optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 5
for e in range(epochs):
    print("[x] Epoch:", e)
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        images.resize_(64, 784)
        output = model.forward(images)
        if output.shape[0] == labels.shape[0]:
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print("[x] mismatch encountered..")
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")

# Anwendung
images_resized = images.resize_(nsize, dim0, dim1)
with torch.no_grad():
    logits = model.forward(images_resized[42])
ps = F.softmax(logits, dim=1)
# die kürzere Version
ps = torch.exp(model(images_resized[42]))

# Anwendung Autograd
aout = np.random.randn(5, 2)
aoutt = torch.from_numpy(aout) + torch.zeros(5, 2, requires_grad=True)
aoutt.retain_grad()
print(aoutt)
y = eval("aoutt**3 + 0.5 * aoutt**2 + aoutt * 2 + 1")
print(y)
y.mean().backward()
print(aoutt.grad)
print((3 * aout ** 2 + 0.5 * 2 * aout + 2) / 10)

# Trainer mit Acc / Validierung
model = classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 30
steps = 0
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        train_accuracy = get_accuracy2(model, trainloader)
        test_accuracy = get_accuracy2(model, testloader)
        print(f'Train Acc: {train_accuracy * 100:.2f}% | Test Acc: {test_accuracy * 100:.2f}%')

# Lernprozedur mit .eval() und .train()
model = classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 30
steps = 0
train_losses, test_losses = [], []
for e in range(epochs):
    print("\n[x] Epoch:", e)
    running_loss = 0
    for images, labels in trainloader:
        model.train()
        optimizer.zero_grad()
        log_ps = model(imgfunk(images))
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        train_accuracy = get_accuracy2(model, trainloader)
        test_accuracy = get_accuracy2(model, testloader)
        diff = np.abs(train_accuracy - test_accuracy)
        print(f'Train Acc: {train_accuracy * 100:.2f}% | Test Acc: {test_accuracy * 100:.2f}% | Diff: {diff * 100:.2f}')

# TODO: Data Augmentation - wie würde das als standard operationen?

# Speichern des Modells
model.state_dict()
torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')
print(state_dict.keys())
# Objekt model kann neu initialisiert werden, also man beginnt wo man verlassen hat
model = Network1() # 784, 10, [400, 200, 100]
model.load_state_dict(state_dict)

# Transfer Learning - für den Teil der Rechennetz für denen keine Updates getan werden
model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

# Beispiel Loop für Training auf CPU
if True:
    device = 'cpu'
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.cpu()
    sum_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        if ii == 10:
            break

# Training auf device
epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            check_model()
            running_loss = 0

# Initialize tokenizer
text = "Hallo Welt, sowas von"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_enc = tokenizer.encode(text)
print(len(text_enc))
trainset = TokenIdsDataset(text_enc, 64)

sampler = RandomSampler(trainset, replacement=True)
dataloader = DataLoader(trainset, batch_size=2, sampler=sampler)
xbatch, ytarget = next(iter(dataloader))
[tokenizer.decode(x) for x in xbatch]
[tokenizer.decode(x) for x in ytarget]


# Testing
input = torch.rand(8, config["context_size"], config["embedding_dim"])
ah = AttentionHead(config)
output = ah(input)
print(output.shape)  # Expected output: torch.Size([8, 256, 64])
mha = MultiHeadAttention(config)
output = mha(input)
print(output.shape)  # Expected output: torch.Size([8, 256, 768])
ff = FeedForward(config)
input = torch.rand(8, config["context_size"], config["embedding_dim"])
output = ff(input)
print(output.shape)  # Expected: torch.Size([8, 256, 768])
b = Block(config)
input = torch.rand(8, config["context_size"], config["embedding_dim"])
output = b(input)
print(output.shape)  # Expected: torch.Size([8, 256, 768])
model = DemoGPT(config).to(device)
output = model(tokenizer.encode("Hi").unsqueeze(dim=0).to(device))
print(output.shape)  # Expected: torch.Size([1, 2, 65])
generate_with_prompt(model, tokenizer, "First Citizen:\n")

# Training Prozedur
batch_size = 64
train_iterations = 500
evaluation_interval = 10
learning_rate = 4e-4
train_split = 0.9

tokenized_text = tokenizer.encode(text).to(device)
ntokens_training = int(train_split * tokenized_text.shape[0])

train_data = tokenized_text[:ntokens_training]
test_data = tokenized_text[ntokens_training:]
train_dataset = TokenIdsDataset(train_data, config["context_size"])
validation_dataset = TokenIdsDataset(test_data, config["context_size"])

train_sampler = RandomSampler(train_dataset, num_samples=batch_size * train_iterations, replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
validation_sampler = RandomSampler(validation_dataset, replacement=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=validation_sampler)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Set up lists to store losses for plotting
train_losses = []
train_steps = []
eval_losses = []
eval_steps = []

for step_num, sample in enumerate(train_dataloader):
    model.train()
    input, targets = sample
    logits = model(input)
    logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
    targets_view = targets.view(batch_size * config["context_size"])
    loss = F.cross_entropy(logits_view, targets_view)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_losses.append(loss.item())
    train_steps.append(step_num)
    print(f"Step {step_num}. Loss {loss.item():.3f}")
    if step_num % evaluation_interval == 0:
        print("Demo GPT:\n" + generate_with_prompt(model, tokenizer, "\n"))
    validation_loss = calculate_validation_loss(model, batches_num=10)
    eval_losses.append(validation_loss)
    eval_steps.append(step_num)
    print(f"Step {step_num}. Validation loss: {validation_loss:.3f}")

cuda_is_available = torch.cuda.is_available()
if cuda_is_available:
    print("All good!")
else:
    print("CUDA is NOT available!")

# https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
prompt = "When people ask an AI Chatbot, they usually ask"
model = "openai-community/gpt2-large"

# Step 1 - Set Up the Text Generation Pipeline
"""
    top_k = 1 === greedy
    sampling=False === greedy
    num_beams = 1 === greedy
"""
generator = pipeline("text-generation", model)
generated_text = generator(prompt, max_length=100, num_return_sequences=1)
generated_text = generator(prompt, max_length=100, num_return_sequences=3, do_sample=True)
generated_text = generator(prompt, max_length=100, num_return_sequences=3, do_sample=False, num_beams=3)
generated_text = generator(prompt, max_length=100, num_return_sequences=3, do_sample=True, num_beams=3)
generated_text = generator(prompt, max_length=100, num_return_sequences=1, do_sample=False, top_k=3)
generated_text = generator(prompt, max_length=100, num_return_sequences=1, do_sample=False, temperature=0.1)
