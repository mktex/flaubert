from importlib import reload
from functools import reduce
import numpy as np
import pandas as pd
import numbers
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay

from flaubert.eda import dfgestalt, dbeschreiben
from flaubert.model.outliers import isoforest
from flaubert.vis import xdiagramme as xdg

reload(xdg)

get_nulls_df_stat = dbeschreiben.get_nulls_df_stat
kateg_werte_liste = dbeschreiben.kateg_werte_liste
frequenz_werte = dbeschreiben.frequenz_werte

import torch.nn.functional as F
import time


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, three=True, dropout=False):
        super(Classifier, self).__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            nn.Dropout(0.4)
        ]
        if not dropout: layers = layers[:3] + layers[4:-1]
        if three:
            layers.extend([
                              nn.Linear(hidden_size, hidden_size),
                              nn.BatchNorm1d(hidden_size),
                              nn.Sigmoid()
                          ] + ([] if not dropout else [nn.Dropout(0.4)]))
        layers.append(nn.Linear(hidden_size, 2))
        print(layers)
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def get_auc(labels, preds):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc_score = auc(fpr, tpr)
    if np.isnan(auc_score):
        print(labels)
        print(preds)
    return fpr, tpr, thresholds, auc_score


def get_perf(model, xloader):
    global batch_size
    def get_perf_batch(xmodel, _batch, _labels):
        ps = F.softmax(xmodel(_batch), dim=1)
        top_p, top_class = ps.topk(1, dim=1)
        predictions = [w[0] for w in top_class.detach().numpy()]
        targets = [int(w) for w in _labels.detach().numpy()]
        fpr, tpr, thresholds, auc_score = get_auc(predictions, targets)
        performance = auc_score
        if len(set(targets)) == 1:
            print(predictions)
            print(targets)
        # print(fpr, tpr)
        # equals = top_class == _labels.view(*top_class.shape)
        # accuracy = torch.mean(equals.type(torch.FloatTensor))
        return performance
    with torch.no_grad():
        model.eval()
        acc_list = []
        for _batch, _labels in iter(xloader):
            if _batch.shape[0] != batch_size or _batch.shape[0] != _labels.shape[0]: continue
            acc_list.append(
                get_perf_batch(model, _batch, _labels)
            )
        res_perf = np.mean(acc_list)
        return res_perf


def get_epoch_validation(model, criterion, xloader):
    with torch.no_grad():
        model.eval()
        running_loss = 0
        for stepk, (xbatch, labels) in enumerate(xloader):
            output = model.forward(xbatch)
            loss = criterion(output, labels)
            running_loss += loss.item()
        return running_loss / len(xloader)


def simple_train_procedure(model, criterion, optimizer, train_loader, val_loader, test_loader,
                           epochs=5, scheduler=None):
    global batch_size
    train_perfs, val_perfs = [], []
    rloss_train, rloss_val = [], []
    for e in range(epochs):
        model.train()
        running_loss = 0
        for stepk, (xbatch, labels) in enumerate(train_loader):
            xbatch = xbatch.to(device)
            labels = labels.to(device)
            if xbatch.shape[0] != batch_size or xbatch.shape[0] != labels.shape[0]: continue
            optimizer.zero_grad()
            output = model.forward(xbatch)
            loss = criterion(output, labels)
            loss.backward()
            current_loss = loss.item()
            optimizer.step()
            running_loss += current_loss
        train_perf = get_perf(model, train_loader)
        val_perf = get_perf(model, val_loader)
        train_perfs.append(train_perf)
        val_perfs.append(val_perf)
        rloss_train.append(running_loss / len(train_loader))
        rv = get_epoch_validation(model, criterion, val_loader)
        rloss_val.append(rv)
        if scheduler is not None: scheduler.step(rv)
        if e % 25 == 0:
            print("[x] Epoch:", e)
            print(f'Train: {train_perf * 100:.2f}% | Val: {val_perf * 100:.2f}% | ' + \
                  f'Loss (train): {rloss_train[-1]:.2f} | Loss (validation): {rloss_val[-1]:.2f}')
        time.sleep(1)
    return model, train_perfs, val_perfs, rloss_train, rloss_val


def evaluate_model(model, data_loader, threshold=.5):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xbatch, labels in data_loader:
            xbatch = xbatch.to(device)
            labels = labels.to(device)
            logits = model(xbatch)
            probs = torch.softmax(logits, dim=1)
            # preds = torch.argmax(logits, dim=1)
            preds = (probs[:, 1] >= threshold).int()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=["No Diabetes", "Diabetes"]))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def vis_auc(model, xloader):
    # fpr, tpr, thresholds = roc_curve(predictions, targets)
    # auc_score = auc(fpr, tpr)
    auc_score, fpr, tpr, thresholds = get_roc_auc_curve_data(model, xloader)
    label=f"Model: AUC: {auc_score}"
    plt.plot(fpr, tpr, '-', linewidth=4, label="Model")
    plt.legend(loc="upper right")
    plt.title(f"Receiver Operating Characteristic: {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    return fpr, tpr, auc_score


def vis_cm(predictions, targets):
    print(classification_report(targets, predictions, zero_division=0.0))
    cm = confusion_matrix(targets, predictions, labels=[0, 1])
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Diabetes", "Diabetes"])
    disp.plot()
    plt.tight_layout()
    plt.grid(False)
    plt.show()
    return cm


def predict_with_labels(xmodel, xloader):
    with torch.no_grad():
        xmodel.eval()
        preds = []
        for _batch, _labels in iter(xloader):
            ps = F.softmax(xmodel(_batch), dim=1)
            top_p, top_class = ps.topk(1, dim=1)
            preds.append([
                top_class.detach().numpy().flatten(),
                _labels.detach().numpy()
            ])
        preds_labels = [list(zip(w[0], w[1])) for w in preds]
        preds_labels = reduce(lambda a, b: a + b, preds_labels)
        preds = [w[0] for w in preds_labels]
        labels = [w[1] for w in preds_labels]
        return preds, labels


def info(model):
    print("\n" + "=" * 80)
    print("[x] Rehennetz:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[x] Anzahl Parametern: {total_params:,}")
    print(f"[x] Für Training: {trainable_params:,}")
    print("[x] Parameter per Layer:")
    for name, param in model.named_parameters():
        print(f"{name:20s} - Shape: {str(param.shape):20s} - Parameters: {param.numel():,}")


def get_roc_auc_curve_data(model, test_loader):
    positive_scores = []
    targets = []
    with torch.no_grad():
        for xbatch, labels in test_loader:
            xbatch = xbatch.to(device)
            logits = model(xbatch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            positive_scores.extend(probs.cpu().numpy())
            targets.extend(labels.numpy())
    fpr, tpr, thresholds = roc_curve(targets, positive_scores)
    auc_score = roc_auc_score(targets, positive_scores)
    return auc_score, fpr, tpr, thresholds


def vis_perflines(train_perfs, val_perfs):
    plt.figure(figsize=(12, 6))
    num_epochs = len(train_perfs)
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, train_perfs, label='Training Performance', linewidth=2, color='#3498db')
    plt.plot(epochs_range, val_perfs, label='Validation Performance', linewidth=2, color='#e74c3c')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title('Training and Validation Performance Over Time', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def vis_loss(rloss_train, rloss_val):
    plt.figure(figsize=(12, 6))
    num_epochs = len(rloss_train)
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, rloss_train, label='Training Loss', linewidth=2, color='#3498db')
    plt.plot(epochs_range, rloss_val, label='Validation Loss', linewidth=2, color='#e74c3c')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def track_experiment(name: str, train_losses: List[float], val_losses: List[float], test_results: Dict[str, float],
                     notes: str = "") -> None:
    """Quelle: Udacity"""
    global experiment_results
    numeric_metrics = {
        k: float(v) for k, v in test_results.items()
        if isinstance(v, numbers.Number)
    }
    experiment_results[name] = {
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'min_val_loss': float(min(val_losses)),
        'loss_gap': float(abs(train_losses[-1] - val_losses[-1])),
        'metrics': numeric_metrics,
        'notes': notes,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    print(f"✓ Experiment '{name}'")


def display_experiment_comparison(sort_by=None, descending: bool = True) -> Optional[pd.DataFrame]:
    """Quelle: Udacity"""
    global experiment_results
    if not experiment_results:
        print("No experiments tracked yet!")
        return None
    all_metric_names = set()
    for res in experiment_results.values():
        all_metric_names.update(res.get('metrics', {}).keys())
    all_metric_names = sorted(all_metric_names)  # stable ordering
    if sort_by is None: sort_by = ["F1"]
    chosen_sort = sort_by[0]
    if sort_by in all_metric_names:
        chosen_sort = sort_by
    elif all_metric_names:
        chosen_sort = all_metric_names[0]  # fallback to first available
    rows = []
    for name, res in experiment_results.items():
        row = {
            'Experiment': name,
            'Val Loss': f"{res['final_val_loss']:.4f}",
            'Loss Gap': f"{res['loss_gap']:.4f}",
        }
        for m in all_metric_names:
            val = res['metrics'].get(m, None)
            row[m.upper() if m.islower() else m] = (f"{val:.4f}" if isinstance(val, numbers.Number) else "")
        row['_sort_val'] = (
            res['metrics'].get(chosen_sort)
            if chosen_sort is not None else res['final_val_loss']
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in list(filter(lambda x: x != 'Experiment', df.columns)):
        df[col] = df[col].astype(np.float32)
    if chosen_sort is not None:
        df = df.sort_values(sort_by, ascending=not descending)
    else:
        df = df.sort_values('Val Loss', ascending=True, key=lambda s: s.astype(float))
    df = df.drop(columns=['_sort_val'])
    return df


def run_one_experiment(named, wide, three, learning_rate, num_epochs):
        print("[x] Experiment:", named, wide, three)
        criterion = nn.CrossEntropyLoss()
        model = Classifier(X_train_scaled.shape[1], wide, three, True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        _ = simple_train_procedure(model, criterion, optimizer, train_loader,
                                   val_loader, test_loader, num_epochs, scheduler)
        model, train_perfs, val_perfs, rloss_train, rloss_val = _
        vis_perflines(train_perfs, val_perfs)
        vis_loss(rloss_train, rloss_val)
        predictions, targets = predict_with_labels(model, test_loader)
        cm = vis_cm(predictions, targets)
        fpr, tpr, auc_score = vis_auc(model, test_loader)
        metrics = evaluate_model(model, test_loader)
        track_experiment(name=named, train_losses=rloss_train, val_losses=rloss_val,
                         test_results={"accuracy": metrics["accuracy"], "precision": metrics["precision"],
                                       "recall": metrics["recall"], "f1": metrics["f1"],
                                       "cm_TP": cm[1][1], "cm_FP": cm[0][1],
                                       "cm_FN": cm[1][0],
                                       "num_epochs": num_epochs, "learning_rate": learning_rate,
                                       "chosen_sort": "RECALL"},
                         notes='.'
        )
        print(display_experiment_comparison())
        print("=" * 60)


# Einstellungen
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data_repository = "./data"

# Input Daten
df = pd.read_csv(f'{data_repository}/diabetes_data.csv')
df = df[list(filter(lambda x: x not in ['AnyHealthcare', 'CholCheck'], df.columns))]
df = df.sample(df.shape[0])
df = df.sample(10000)
target = "Diabetes_binary"

do_enhance_features = True
do_outlier_detection = True
do_missing_values = True
do_visualisierungen = True

if do_outlier_detection: df = isoforest.do_outlier_detection(df, target)

if do_missing_values:
    missings = dbeschreiben.get_nulls_df_stat(df, return_all=True)
    assert missings['nullen'].sum() == 0, "Es gibt fehlende Werte in Dataframe"

if do_visualisierungen:
    xdg.vis_binary_barplot(df, target)
    xdg.vis_grid_histograms(df, ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'GenHlth'])
    xdg.vis_corr_as_barplot(df, target)

if do_enhance_features:
    df = dfgestalt.do_enhance_features(df, target)

_ = dfgestalt.prepare_training_datasets(df, target)
X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, xplain_most = _
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_val_tensor = torch.FloatTensor(X_val_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train.values)
y_val_tensor = torch.LongTensor(y_val.values)
y_test_tensor = torch.LongTensor(y_test.values)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

experiment_results = {}
run_one_experiment("baseline", 64, True, 0.001, 100)

if True:
    named = "final choice"
    wide, three = 128, False
    num_epochs = 100
    learning_rate = 0.0008
    print("[x] Experiment:", named, wide, three)
    criterion = nn.CrossEntropyLoss()
    model = Classifier(X_train_scaled.shape[1], wide, three, True).to(device)
    info(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    _ = simple_train_procedure(model, criterion, optimizer, train_loader, val_loader, test_loader, num_epochs, scheduler)
    model, train_perfs, val_perfs, rloss_train, rloss_val = _
    vis_perflines(train_perfs, val_perfs)
    vis_loss(rloss_train, rloss_val)
    predictions, targets = predict_with_labels(model, test_loader)
    cm = vis_cm(predictions, targets)
    fpr, tpr, auc_score = vis_auc(model, test_loader)
    metrics = evaluate_model(model, test_loader, 0.45)
    track_experiment(
        name=named,
        train_losses=rloss_train,
        val_losses=rloss_val,
        test_results={"accuracy": metrics["accuracy"],
                      "precision": metrics["precision"],
                      "recall": metrics["recall"],
                      "f1": metrics["f1"],
                      "cm_TP": cm[1][1],
                      "cm_FP": cm[0][1],
                      "cm_FN": cm[1][0],
                      "num_epochs": num_epochs,
                      "learning_rate": learning_rate,
                      "chosen_sort": "RECALL"
                      },
        notes='.'
    )

display_experiment_comparison(["F1", "RECALL"])





