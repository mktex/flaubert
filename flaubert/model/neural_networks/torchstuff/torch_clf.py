from flaubert.model.neural_networks.torchstuff.tutils import *


def get_simple_stats(xreview):
    nchars = len(list(filter(lambda x: x != " ", xreview)))
    nwords = len(xreview.split(" "))
    return nchars, nwords


def add_stats(df):
    simple_stats_list = list(map(lambda xreview: get_simple_stats(xreview), df['text'].values))
    df["nchars"] = [w[0] for w in simple_stats_list]
    df["nwords"] = [w[1] for w in simple_stats_list]
    return df


def info(df):
    print("\n", df.info(), "\n")
    print("\n", df.describe(), "\n")
    return df


def vis():
    print("\n[x] Positiv:")
    record = train_df[train_df['label'] == 1].sample(10).iloc[5]
    print(record)
    print(record['text'])
    print("\n[x] Negativ:")
    record = train_df[train_df['label'] == 0].sample(10).iloc[5]
    print(record)
    print(record['text'])
    # Counts
    if True:
        t1 = train_df.groupby('label').count()
        t2 = test_df.groupby('label').count()
        t1.plot(kind="bar", color="cyan", label="train")
        t2.plot(kind="bar", color="cyan", label="test")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.title("Train Counts")
        plt.show()
    # Number of characters
    if True:
        train_df["nchars"].hist(bins=45, color="black", label="train")
        test_df["nchars"].hist(bins=45, color="gray", alpha=0.5, label="test")
        plt.title("Histogram der Wortlängen")
        plt.legend()
        plt.tight_layout()
        plt.show()


def get_prep_data_training():
    global train_data, train_df, val_data
    train_size = int(0.9 * len(train_df))
    shuffled_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_data = shuffled_df.iloc[:train_size]
    val_data = shuffled_df.iloc[train_size:]
    return train_data, val_data


def schritt_fuer_schritt_ausbau():
    global model, logits, loss, batch
    attention_head = AttentionHead(config).to(device)
    dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)
    attention_output = attention_head(dummy_input)
    print("[x] AttentionHead output shape:", attention_output.shape)
    multi_head_attention = MultiHeadAttention(config).to(device)
    dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)
    mha_output = multi_head_attention(dummy_input)
    print("[x] MultiHeadAttention output shape:", mha_output.shape)
    feed_forward = FeedForward(config).to(device)
    dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)
    ff_output = feed_forward(dummy_input)
    print("[x] FeedForward output shape:", ff_output.shape)
    block = Block(config).to(device)
    dummy_input = torch.randn(BATCH_SIZE, config["context_size"], config["d_embed"]).to(device)
    block_output = block(dummy_input)
    print("[x] Block output shape:", block_output.shape)
    model = DemoGPT(config, _output_logits=True, _output_size=config["num_classes"]).to(device)
    dummy_token_ids = torch.randint(0, config["vocabulary_size"], (BATCH_SIZE, config["context_size"])).to(device)
    logits = model(dummy_token_ids)
    print("[x] DemoGPT output shape:", logits.shape)
    print("[x] Logits:\n", logits[:3])
    assert logits.size(1) == config["num_classes"], (
        f"Expected number of classes {config['num_classes']}, "
        f"but got {logits.size(1)}"
    )
    assert logits.size(0) == BATCH_SIZE, (
        f"Expected batch size {BATCH_SIZE}, "
        f"but got {logits.size(0)}"
    )
    return model


@torch.no_grad()
def get_epoch_validation():
    global model, val_loader
    with torch.no_grad():
        model.eval()
        rloss = 0
        for stepk, (_xbatch, _labels) in enumerate(val_loader):
            if _xbatch.shape[0] != BATCH_SIZE: continue
            _logits = model.forward(_xbatch)
            _loss = F.cross_entropy(_logits, _labels)
            rloss += _loss.item()
        return rloss / len(val_loader)


@torch.no_grad()
def calculate_accuracy(xloader):
    global model
    model.eval()
    xiter = iter(xloader)
    bootstrap_means = []
    for _batch, _targets in iter(xiter):
        _logits = model(_batch.to(device))
        probs = F.softmax(_logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        acc = np.mean(preds.to("cpu").numpy() == _targets.to("cpu").numpy()) * 100
        bootstrap_means.append(acc)
    accuracy = np.mean(bootstrap_means)
    return accuracy


@torch.no_grad()
def checke_ein_datensatz_aus_training():
    global model, train_data, MAX_LENGTH
    print("=" * 80)
    print("[x] Beispiel Voraussage..")
    one_example_df = train_data.sample(1)
    one_example, one_label = next(iter(DataLoader(
        IMDBDataset(one_example_df, tokenizer, max_length=MAX_LENGTH), batch_size=1, shuffle=False)
    ))
    model.eval()
    _logits = model(one_example)
    probs = F.softmax(_logits, dim=-1)
    preds = torch.argmax(probs, dim=1)
    print(one_example_df['text'].iloc[0], "\n")
    print("[x] logits:", _logits)
    print("[x] Probs:", probs)
    print("[x] Voraussage:", preds)
    print("[x] Erwartet:", one_label)
    print(f"[x] Match? {'JA' if preds[0].to(device).numpy() == one_label.to(device).numpy() else 'NEIN'}")


# Quelle: https://ai.stanford.edu/~amaas/data/sentiment/
ds = datasets.load_dataset("stanfordnlp/imdb")
train_df = add_stats(pd.DataFrame(ds["train"]))
test_df = add_stats(pd.DataFrame(ds["test"]))
del ds

# Simple Statistiken
info(train_df)
info(test_df)
vis()

# Training/Test
train_data, val_data = get_prep_data_training()

# Einstellungen fürs Training
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
XPLORE_PERCENTAGE = 0.2
NITER_PER_EPOCH = int(XPLORE_PERCENTAGE * EPOCHS * (train_data.shape[0] // BATCH_SIZE))
print(f"[x] Ausführung mit {EPOCHS} Epochen und {NITER_PER_EPOCH} Iterationen per Epoch")

# Anwendung vordefinierter Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sample_texts = train_data['text'].sample(1, random_state=42).tolist()
tokenized_samples = tokenizer(sample_texts, truncation=True, padding="max_length",
                              max_length=MAX_LENGTH, return_tensors="pt")
train_dataset = IMDBDataset(train_data, tokenizer, max_length=MAX_LENGTH)
val_dataset = IMDBDataset(val_data, tokenizer, max_length=MAX_LENGTH)
test_dataset = IMDBDataset(test_df, tokenizer, max_length=MAX_LENGTH)
assert val_dataset[0][0].shape[0] == MAX_LENGTH

# Dataloaders
train_sampler = RandomSampler(train_dataset, num_samples=BATCH_SIZE*NITER_PER_EPOCH, replacement=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
# xbatch, ytarget = next(iter(train_loader))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

config = {
    "vocabulary_size": tokenizer.vocab_size,  # e.g., ~30522 for bert-base-uncased
    "num_classes": 2,                         # binary classification (pos/neg)
    "d_embed": 128,
    "context_size": MAX_LENGTH,
    "layers_num": 4,
    "heads_num": 4,
    "dropout_rate": 0.1,
    "use_bias": True
}
config["head_size"] = config["d_embed"] // config["heads_num"]
pprint(config)

model = schritt_fuer_schritt_ausbau()
# Dasselbe: model = DemoGPT(config, _output_logits=True, _output_size=config["num_classes"]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003)

validation_accuracy = calculate_accuracy(val_loader)
print(f"[x] Accuracy @ Start: {validation_accuracy:.2f}%")

# Training Prozedur
for epoch in range(EPOCHS):
    print("=" * 80)
    print(f"[x] Epoch {epoch}")
    print_atk = int(0.1 * len(train_loader))
    running_loss, loss_counter = 0.0, 0
    for step, (batch, labels) in enumerate(train_loader):
        _ = model.train()
        batch_size = batch.shape[0]
        if batch_size != BATCH_SIZE: continue
        batch = batch.to(device)
        labels = labels.to(device)
        logits = model(batch)
        logits_view = logits.view(batch_size, config["num_classes"])
        targets_view = labels.view(batch_size)
        loss = F.cross_entropy(logits_view, targets_view)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_counter += 1
        if (step + 1) % print_atk == 0:
            train_loss = running_loss / loss_counter
            val_loss = get_epoch_validation()
            loss_gap = np.abs(train_loss - val_loss)
            print(f"\t {100*(step + 1)/len(train_loader):.0f}% | Loss (train): {train_loss:.4f} " +
                  f"| Loss (val): {val_loss:.4f} | Abweichung {loss_gap:.4f}")
            running_loss, loss_counter = 0.0, 0
            checke_ein_datensatz_aus_training()
            time.sleep(1)  # release cpu heat
    print(f"\n[x] Epoch {epoch} Training beendet, Validierung folgt..")
    val_accuracy = calculate_accuracy(val_loader)
    print(f"[x] Accuracy (validation): {val_accuracy:.2f}%")

test_accuracy = calculate_accuracy(test_loader)
print(f"[x] Accuracy (Test): {test_accuracy:.2f}%")
