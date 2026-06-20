import time
from importlib import reload

import pandas as pd
from sklearn.datasets import make_blobs

from flaubert.eda import dfgestalt
from flaubert.model.neural_networks.torchstuff import tutils
reload(tutils)
from flaubert.model.neural_networks.torchstuff.tutils import *

# Einfache Beispiele
beispiel_generator()
beispiel_for_schleife_mit_else()
beispiel_text_laden()
beispiele_torch()

# verschiedene Netzarchitekturen
model = Network1()

# Beispiel Datasets -> DataFrame -> PyTorch Tensors
X_tensor, y_tensor = beispiel_dataset2dataframe()
# dataset = datasets.load_dataset('Praxash1/AIgrow', split='train')
# dataset = datasets.load_dataset("MongoDB/whatscooking.restaurants", split="train")
# dataset = datasets.load_dataset('Redsmoothy/HR_Attrition', split='train')

# Data Loaders
# trainloader, testloader, train_data, test_data = beispiel_cat_dog_data()
info_dataloader()
trainloader, trainset = beispiel_bild_laden_mit_transformer(ttrial=3, nrecords_per_batch=32)
rand_train_dataloader = beispiel_data_loader_mit_stichproben(trainset, 64, 100)
trainloader, testloader, train_data, test_data = beispiel_dataloader_bilder_im_ordner()

# Numpy -> Dataloader
data, labels = make_blobs(n_samples=100, n_features=10, centers=4)
df = pd.DataFrame(data, columns=[f"feat{w}" for w in range(data.shape[1])])
df["target"] = labels
_ = dfgestalt.prepare_training_datasets(df, "target")
X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, xplain_most = _
trainloader = from_numpy_xtrain_to_pytorch_dataloader(X_train_scaled, y_train, batch_size=32)

# Tokenizers
beispiel_tokenizer(text="Hallo Welt, sowas von")

# Ableitungen
beispiel_anwendung_autograd()

# Beispiel Sequential
model, input_size, trainloader, trainset = beispiel_sequential()

# Transferlearning
model = beispiel_transfer_learning()

# Trainingprozeduren
beispiel_optimization_one_step()

# Einfache Trainingprozedur auf MNIST
model, batch_size, h, w, trainloader = simple_train_procedure_mnist()

# Pipelines
beispiel_pipeline()



