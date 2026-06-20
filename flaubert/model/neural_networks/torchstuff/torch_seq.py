from importlib import reload
import time

from flaubert.model.neural_networks.torchstuff import tutils
reload(tutils)
from flaubert.model.neural_networks.torchstuff.tutils import *


def test_tokenizer(xstr="Hallo Welt"):
    print("[x] Input Text:", xstr)
    print("[x] => ", tokenizer.encode(xstr))
    print("[x] Die Umkehrung, Dasselbe wie: |{tokenizer.decode(tokenizer.encode(xstr))}|")
    print(f"[x] Wörterbuchsgröße (Anzahl Symbole): {tokenizer.vocabulary_size()}")


# Mit Tokenizer, Transformer
dset = datasets.load_dataset('Trelis/tiny-shakespeare')
df = pd.DataFrame(dset['train'])
text = '\n'.join(df["Text"].values)
tokenizer = CharTokenizer.train_from_text(text)
test_tokenizer(df["Text"].sample().iloc[0].split("\n")[0])

config = {
    "vocabulary_size": tokenizer.vocabulary_size(),
    "d_embed": 128,
    "context_size": 128,
    "layers_num": 4,
    "heads_num": 4,
    "dropout_rate": 0.1,
    "use_bias": True
}
config["head_size"] = config["d_embed"] // config["heads_num"]
if True:
    xinput = torch.rand(8, config["context_size"], config["d_embed"])
    ah = AttentionHead(config); output = ah(xinput); print(output.shape)
    mha = MultiHeadAttention(config); output = mha(xinput); print(output.shape)
    xinput = torch.rand(8, config["context_size"], config["d_embed"])
    ff = FeedForward(config); output = ff(xinput); print(output.shape)
    xinput = torch.rand(8, config["context_size"], config["d_embed"])
    b = Block(config); output = b(xinput)
    print(output.shape)
    model = DemoGPT(config).to(device)
    output = model(tokenizer.encode("Hi").unsqueeze(dim=0).to(device)); print(output.shape)
    generate_with_prompt(model, config, tokenizer, "First Citizen:\n")

# Einstellungen Training Prozedur
batch_size = 64
train_iterations = 500
evaluation_interval = 10
learning_rate = 4e-4
train_split = 0.9
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Datasets mit tokenizer
tokenized_text = tokenizer.encode(text).to(device)
ntokens_training = int(train_split * tokenized_text.shape[0])
train_data = tokenized_text[:ntokens_training]
test_data = tokenized_text[ntokens_training:]
train_dataset = TokenIdsDataset(train_data, config["context_size"])
validation_dataset = TokenIdsDataset(test_data, config["context_size"])

# Loaders mit RandomSampler
train_sampler = RandomSampler(train_dataset, num_samples=batch_size * train_iterations, replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
validation_sampler = RandomSampler(validation_dataset, replacement=True)
val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

# Training Prozedur
train_losses = []
train_steps = []
eval_losses = []
eval_steps = []
for step_num, sample in enumerate(train_dataloader):
    _ = model.train()
    xinputs, targets = sample
    logits = model(xinputs)
    logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
    targets_view = targets.view(batch_size * config["context_size"])
    loss = F.cross_entropy(logits_view, targets_view)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_losses.append(loss.item())
    train_steps.append(step_num)
    if step_num % evaluation_interval == 0:
        print("Demo GPT:\n" + generate_with_prompt(model, config, tokenizer, "\nWhat say you, mylord?"))
    validation_loss = calculate_validation_loss(model, 10, batch_size, val_dataloader, config)
    eval_losses.append(validation_loss)
    eval_steps.append(step_num)
    print(f"Step {step_num}. Loss {loss.item():.3f} | Validation loss: {validation_loss:.3f}")
    time.sleep(0.5)


"""
import os
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
from IPython.display import display
import ipywidgets as widgets
%matplotlib inline

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

plot_output = widgets.Output()
display(plot_output)
"""











