import os, time
from pprint import pprint
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoTokenizer
from transformers import pipeline
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from torchvision import datasets as tdatasets, transforms, models

device = "cpu"
cuda_is_available = torch.cuda.is_available()

xfclean = lambda: os.system('clear')

def beispiel_generator():
    def iclock():
        ik = 1
        while ik <= 4:
            yield ik
            ik += 1
            if ik == 5: ik = 1
    xc = iclock()
    next(xc)


def beispiel_for_schleife_mit_else():
    """ Beispiel """
    for i in range(5):
        print(i)
    else:
        print("Loop just finished!")


def beispiele_torch():
    print("[x] Basis OPs..")
    torch.manual_seed(7)
    features = torch.randn((1, 3))
    torch.rand(5)
    torch.rand(5, 5)  # 5x5 matrize
    torch.ones(5)
    torch.ones(5, 5)
    torch.arange(10)
    torch.tril(torch.rand(5, 5))  # triangular matrix, lower != 0 rest 0
    torch.cat((torch.rand(5, 5), torch.rand(5, 5)), axis=0)  # catenate on rows
    x = torch.arange(10)
    x_star = x.view(5, 2)
    x_star.view(10)
    a = torch.ones(5)
    b = a.numpy()
    a = np.array([1, 2, 3])
    t = torch.from_numpy(a)


def beispiel_dataset2dataframe():
    """ Quelle: Udacity Materialien
        https://huggingface.co/datasets
    """
    # lade einen Dataset und bringe ins Pandas DataFrame Format
    dset = datasets.load_dataset('Pablinho/movies-dataset', split='train')
    df = pd.DataFrame(dset)
    df['Vote_Average'] = pd.to_numeric(df['Vote_Average'], errors='coerce')
    df_clean = df[(df['Vote_Average'] > 0) & (df['Popularity'] > 0) & (df['Vote_Average'].notna())].copy()
    df_subset = df_clean.sample(n=500, random_state=42)
    vote_avg = df_subset['Vote_Average'].values
    popularity = df_subset['Popularity'].values
    vote_avg_norm = (vote_avg - vote_avg.min()) / (vote_avg.max() - vote_avg.min())
    popularity_norm = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    labels = ((vote_avg > 7.0) & (popularity > 50)).astype(int)
    # Übergang ins PyTorch
    X = np.column_stack([vote_avg_norm, popularity_norm])
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(labels).unsqueeze(1)
    return X_tensor, y_tensor


def beispiel_text_laden(xfiles_path="./data/input_text.txt"):
    def xtract_content(subdir, xfile):
        with open(f'{subdir}/{xfile}'.replace('//', '/'), 'r') as f:
            one_file_content = f.read()
        return one_file_content
    subdir, dirs, files = list(os.walk(xfiles_path))[0]
    txt_files = list(filter(lambda x: '.txt'==x[-4:], files))
    print(f"[x] Extracting from {xfiles_path} {len(txt_files)} files..")
    return list(map(lambda fpath: xtract_content(subdir, fpath), txt_files))


def beispiel_bild_laden_mit_transformer(ttrial=1, nrecords_per_batch=32):
    if ttrial == 1:
        transform = transforms.Compose([transforms.Resize(64),
                                        transforms.CenterCrop(63),
                                        transforms.ToTensor()])
    elif ttrial == 2:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        transforms.CenterCrop(63)
                                        ])
    elif ttrial == 3:
        """
            transforms.Resize(128),
            transforms.RandomResizedCrop(127),        
        """
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Normalize((0.5, ), (0.5, )),
                                        transforms.RandomRotation(15),
                                        transforms.Resize(64),
                                        transforms.CenterCrop(63)
                                        ])
    else:
        print("[x] ttrial darf: 1, 2 oder 3 sein!")
        return
    trainset = tdatasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=nrecords_per_batch, shuffle=True)
    images, labels = next(iter(trainloader))
    visimgs = images[10].squeeze()
    print(visimgs, visimgs.shape)
    plt.imshow(visimgs.numpy().squeeze(), cmap='Greys_r')
    plt.show()
    return trainloader, trainset


def beispiel_sequential():
    trainloader, trainset = beispiel_bild_laden_mit_transformer(1, 32)
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.Softmax(dim=1))
    images, labels = next(iter(trainloader))
    images.resize_(images.shape[0], 1, 784)
    ps = model.forward(images[0, :])
    plt.imshow(images[0].reshape(28, 28).numpy().squeeze(), cmap='Greys_r')
    plt.show()
    return model, input_size, trainloader, trainset


def beispiel_transfer_learning():
    """
        Transfer Learning - für den Teil der Rechennetz für denen keine Updates getan werden
    """
    model = models.densenet121(pretrained=True)
    for param in model.parameters(): param.requires_grad = False
    classifier = nn.Sequential(
        OrderedDict([
        ('fc1', nn.Linear(1024, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    return model


def info_dataloader():
    # Bilder handeln mit DataLoader
    dataloader, trainset = beispiel_bild_laden_mit_transformer(ttrial=1, nrecords_per_batch=64)
    print("""
        images, labels = next(iter(dataloader))
    """)
    images, labels = next(iter(dataloader))
    print(images)
    print(labels)
    print("""
    Für Loop:
        for images, labels in dataloader: pass
    """)


def beispiel_data_loader_mit_stichproben(xdataset, batch_size, niterations):
    train_sampler = RandomSampler(xdataset, num_samples=batch_size * niterations, replacement=True)
    rand_train_dataloader = DataLoader(xdataset, batch_size=batch_size, sampler=train_sampler)
    return rand_train_dataloader


def beispiel_dataloader_bilder_im_ordner():
    print("[x] die Bilder fehlen, Code beispielhaft.")
    data_dir = './data/Cat_Dog_data/'
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
    return trainloader, testloader, train_data, test_data


def beispiel_anwendung_autograd():
    aout = np.random.randn(5, 2)
    aoutt = torch.from_numpy(aout) + torch.zeros(5, 2, requires_grad=True)
    aoutt.retain_grad()
    print(aoutt)
    y = eval("aoutt**3 + 0.5 * aoutt**2 + aoutt * 2 + 1")
    print(y)
    y.mean().backward()
    print(aoutt.grad)
    print((3 * aout ** 2 + 0.5 * 2 * aout + 2) / 10)


def beispiel_optimization_one_step():
    trainloader, trainset = beispiel_bild_laden_mit_transformer(ttrial=3, nrecords_per_batch=32)
    images, labels = next(iter(trainloader))
    nbatches, h, w = images.squeeze().shape
    images = images.view(nbatches, h * w)
    model = simple_sequential_3(h * w)
    criterion = nn.NLLLoss()
    print('[x] Gewichtungen initial - ', model[0].weight)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    print("[x] Gradient (vor dem Lernen):", model[0].weight.grad)
    # Voraussagen: with torch.no_grad(): logits = model(images)
    logits = model(images)
    loss = criterion(logits, labels)
    # equivalent: logits_forward = model.forward(images)
    # equivalent: loss_forward = criterion(logits_forward, labels)
    loss.backward()
    optimizer.step()
    print("[x] Loss:", loss)
    print("[x] Gradient:", model[0].weight.grad)
    print('[x] Gewichtungen nach Optimierung ', model[0].weight)
    return model, nbatches, h, w, trainloader


def beispiel_pipeline():
    # https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
    prompt = "When people ask an AI Chatbot, they usually ask"
    model_name = "openai-community/gpt2-large"
    """
        top_k = 1 === greedy
        sampling=False === greedy
        num_beams = 1 === greedy
    """
    generator = pipeline("text-generation", model_name)
    generated_text = generator(prompt, max_length=100, num_return_sequences=1)
    print(generated_text)
    generated_text = generator(prompt, max_length=100, num_return_sequences=3, do_sample=True)
    print(generated_text)
    generated_text = generator(prompt, max_length=100, num_return_sequences=3, do_sample=False, num_beams=3)
    print(generated_text)
    generated_text = generator(prompt, max_length=100, num_return_sequences=3, do_sample=True, num_beams=3)
    print(generated_text)
    generated_text = generator(prompt, max_length=100, num_return_sequences=1, do_sample=False, top_k=3)
    print(generated_text)
    generated_text = generator(prompt, max_length=100, num_return_sequences=1, do_sample=False, temperature=0.1)
    print(generated_text)


def from_numpy_xtrain_to_pytorch_dataloader(x_train, y_train, batch_size=32):
    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size
    )
    return trainloader


def beispiel_tokenizer(text="Hallo Welt, sowas von"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_enc = tokenizer.encode(text)
    print(len(text_enc))
    trainset = TokenIdsDataset(text_enc, 2)
    sampler = RandomSampler(trainset, replacement=True)
    dataloader = DataLoader(trainset, batch_size=2, sampler=sampler)
    xbatch, ytarget = next(iter(dataloader))
    decoded_sequence_batch = [tokenizer.decode(x) for x in xbatch]
    decoded_sequence_targets = [tokenizer.decode(x) for x in ytarget]
    for elem in zip(decoded_sequence_batch, decoded_sequence_targets):
        print(elem)


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


def softmax(z):
    return torch.exp(z) * torch.div(1, torch.sum(torch.exp(z), axis=1)).reshape(-1, 1)


def apply_sigmoid(x):
    return torch.sigmoid(x)


def apply_tanh(x):
    return torch.tanh(x)


def apply_relu(x):
    return torch.relu(x)


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


class Network3(nn.Module):
    def __init__(self):
        super(Network3, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x


class Network4(nn.Module):
    def __init__(self):
        super(Network4, self).__init__()
        self.layer1 = nn.Linear(12, 20)
        self.layer2 = nn.Linear(10, 5)
        self.layer3 = nn.Linear(5, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def print_parameter_breakdown(model):
    for idx, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            weight_params = layer.weight.numel()
            bias_params = layer.bias.numel()
            print(f"  Layer {idx}: {layer}")
            print(f"    → Weight shape: {layer.weight.shape} = {weight_params:,} parameters")
            print(f"    → Bias shape:   {layer.bias.shape} = {bias_params:,} parameters")


def simple_sequential_1():
    input_size, output_size = 5, 1
    hidden_size_wide = 512
    model_wide = nn.Sequential(
        nn.Linear(input_size, hidden_size_wide),  # 5 → 512
        nn.ReLU(),
        nn.Linear(hidden_size_wide, output_size),  # 512 → 1
        nn.Sigmoid()
    )
    nparams = count_parameters(model_wide)
    print_parameter_breakdown(model_wide)
    print(f"[x] Anzahl Parametern: {nparams:,}")
    return model_wide


def simple_sequential_2():
    input_size, output_size = 5, 1
    hidden_size_deep = 32
    model_deep = nn.Sequential(
        nn.Linear(input_size, hidden_size_deep),  # 5 → 32
        nn.ReLU(),
        nn.Linear(hidden_size_deep, hidden_size_deep),  # 32 → 32
        nn.ReLU(),
        nn.Linear(hidden_size_deep, hidden_size_deep),  # 32 → 32
        nn.ReLU(),
        nn.Linear(hidden_size_deep, output_size),  # 32 → 1
        nn.Sigmoid()
    )
    nparams = count_parameters(model_deep)
    print_parameter_breakdown(model_deep)
    print(f"[x] Anzahl Parametern: {nparams:,}")
    return model_deep


def simple_sequential_3(input_size):
    model = nn.Sequential(nn.Linear(input_size, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))
    nparams = count_parameters(model)
    print_parameter_breakdown(model)
    print(f"[x] Anzahl Parametern: {nparams:,}")
    return model


def simple_sequential_4():
    _model = nn.Sequential(nn.Linear(784, 128),
                           nn.ReLU(),
                           nn.Linear(128, 64),
                           nn.ReLU(),
                           nn.Linear(64, 10),
                           nn.Dropout(p=0.2),
                           nn.LogSoftmax(dim=1))
    return _model


def nn_haendisch():
    features = torch.randn((1, 3))
    n_input = features.shape[1]
    n_hidden = 2
    n_output = 1
    W1 = torch.randn(n_input, n_hidden)
    W2 = torch.randn(n_hidden, n_output)
    B1 = torch.randn((1, n_hidden))
    B2 = torch.randn((1, n_output))
    hout = activation(torch.mm(features, W1) + B1)
    out = activation(torch.mm(hout, W2) + B2)
    print(f"""
    Beispielhaft..    
        Features: {features}
        n_input, n_hidden, n_output: {n_input} | {n_hidden} | {n_output}
        => W1, b1: {W1}, {B1}
        => W2, b2: {W2}, {B2}
        => hout: sigma(W1 @ x.T + b1) {hout}
        => out: sigma(W2 @ hout.T + b2) {out}
    """)
    return hout, out


def get_accuracy(model, xloader, imgfunk):
    def get_accuracy_batch(xmodel, _images, _labels):
        ps = torch.exp(xmodel(imgfunk(_images)))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == _labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        return accuracy.item()
    with torch.no_grad():
        model.eval()
        acc_list = []
        for _images, _labels in iter(xloader):
            if np.random.rand() > 0.35: continue
            acc_list.append(
                get_accuracy_batch(model, _images, _labels)
            )
        res_accuracy = np.mean(acc_list)
        model.train()
        return res_accuracy


def model_speichern(model, model_loader_method, args):
    model.state_dict()
    torch.save(model.state_dict(), 'checkpoint.pth')
    state_dict = torch.load('checkpoint.pth')
    print(state_dict.keys())
    model = model_loader_method(*args)
    model.load_state_dict(state_dict)
    return model


def simple_train_procedure_mnist():
    model, batch_size, h, w, trainloader = beispiel_optimization_one_step()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    imgfunk = lambda images: images.view(batch_size, h * w)
    if True:
        epochs = 5
        for e in range(epochs):
            print("[x] Epoch:", e)
            running_loss = 0
            print_every_step = 100
            for stepk, (images, labels) in enumerate(trainloader):
                # on device: images, labels = images.to(device), labels.to(device)
                images = images.view(batch_size, h * w)
                _ = images.resize_(batch_size, h * w)
                output = model.forward(images)
                if output.shape[0] == labels.shape[0]:
                    optimizer.zero_grad()
                    loss = criterion(output, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    running_loss += loss.item()
                else:
                    print("[x] mismatch encountered..")
                if stepk % print_every_step == 0:
                    print(f"Training loss: {running_loss / len(trainloader)}")
                    train_accuracy = get_accuracy(model, trainloader, imgfunk)
                    # test_accuracy = get_accuracy2(model, testloader)
                    print(f'Train Acc: {train_accuracy * 100:.2f}%')
                    running_loss = 0
                    time.sleep(1)
    model_speichern(model, simple_sequential_3, args=(h * w,))


def simple_train_procedure(model, train_loader, val_loader, test_loader):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 5
    for e in range(epochs):
        print("[x] Epoch:", e)
        running_loss = 0
        print_every_step = 100
        for stepk, (xbatch, labels) in enumerate(train_loader):
            output = model.forward(xbatch)
            if output.shape[0] == labels.shape[0]:
                optimizer.zero_grad()
                loss = criterion(output, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            else:
                print("[x] mismatch encountered..")
            if stepk % print_every_step == 0:
                print(f"[x] training loss: {running_loss / len(train_loader)}")
                train_accuracy = get_accuracy(model, train_loader)
                val_accuracy = get_accuracy(model, val_loader)
                print(f'Train Acc: {train_accuracy * 100:.2f}% | Val Acc: {val_accuracy * 100:.2f}%')
                running_loss = 0
                time.sleep(1)
    model_accuracy = get_accuracy(model, test_loader)
    return model, model_accuracy


def apply_classifier(xmodel, _images):
    return torch.exp(xmodel(imgfunk(_images)))


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
        Q = self.Q_weights(input)  # (B, T, head_size)
        K = self.K_weights(input)  # (B, T, head_size)
        V = self.V_weights(input)  # (B, T, head_size)
        attention_scores = Q @ K.transpose(1, 2)  # Q @ K^T => (B, T, T)
        attention_scores = attention_scores.masked_fill(
            self.causal_attention_mask[:tokens_num, :tokens_num] == 0, -torch.inf
        )  # Causal Mask
        attention_scores = attention_scores / (K.shape[-1] ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)
        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config["dropout_rate"])
    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]
        scores_change = torch.cat(heads_outputs, dim=-1)  # (B, T, heads_num * head_size)
        scores_change = self.linear(scores_change)  # (B, T, d_embed)
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
        norm1 = self.layer_norm_1(input)
        x = self.multi_head(norm1)
        x = x + residual
        residual = x
        norm2 = self.layer_norm_2(x)
        x = self.feed_forward(norm2)
        return x + residual


class DemoGPT(nn.Module):
    output_logits = False
    output_size = None
    def __init__(self, config,
                 _output_logits=False, _output_size=None):
        super().__init__()
        self.output_logits = _output_logits
        self.output_size = config["vocabulary_size"] if _output_size is None else _output_size
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])
        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(config["d_embed"])
        self.unembedding = nn.Linear(config["d_embed"], self.output_size, bias=False)
    def forward(self, token_ids):
        batch_size, tokens_num = token_ids.shape
        x = self.token_embedding_layer(token_ids)
        sequence = torch.arange(tokens_num, device=device)
        x = x + self.positional_embedding_layer(sequence).unsqueeze(0)
        x = self.layers(x)
        x = self.layer_norm(x)
        x = self.unembedding(x)
        if self.output_logits: x = torch.mean(x, dim=1)
        return x


def generate(model, config, prompt_ids, max_tokens):
    output_ids = prompt_ids
    for _ in range(max_tokens):
        if output_ids.shape[1] >= config["context_size"]:
            break
        with torch.no_grad():
            logits = model(output_ids)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)
    return output_ids


def generate_with_prompt(model, config, tokenizer, prompt, max_tokens=100):
    model.eval()
    prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to(device)
    return tokenizer.decode(generate(model, config, prompt, max_tokens)[0])


@torch.no_grad()
def calculate_validation_loss(model, batches_num, batch_size, test_dataloader, config):
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


class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1000):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        input_text = self.data['text'].iloc[idx]
        label = self.data['label'].iloc[idx]
        return self.tokenizer(input_text,
                              truncation=True, padding="max_length",
                              max_length=self.max_length,
                              return_tensors="pt")['input_ids'][0, :], label





