from pathlib import Path
import glob
import utils
import numpy as np
import torch
import pandas as pd
from sound_mel_nn import MelSoundDataset
from sklearn import metrics
import json
from dataclasses import dataclass
import os
import pickle
import seaborn as sns
from stat_nn import StatDataset
from imblearn.metrics import geometric_mean_score

CWD = Path.cwd()
DEVICE = 0


def print_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    lines = (
        "           Real      \n"
        + "          1      0\n"
        + "      ---------------\n"
        + f"    1    {tp:>4}|{fp:<4}\n"
        + "pred  ---------------\n"
        + f"    0    {fn:>4}|{tn:<4}\n"
        + "      ---------------\n"
    )
    return lines


def infer_nn(model, dataset_and_labels_per_file):
    model.eval()
    model.to(DEVICE)
    y_pred = np.zeros(len(dataset_and_labels_per_file))
    y_true = np.zeros(len(dataset_and_labels_per_file))
    for i, (file_features, label) in enumerate(dataset_and_labels_per_file):
        with torch.no_grad():
            features = torch.from_numpy(file_features).to(DEVICE)
            y_pred[i] = model(features).cpu().numpy().mean()
            y_true[i] = label
    return y_pred, y_true


def infer_ae(model, dataset_and_labels_per_file):
    model.eval()
    model.to(DEVICE)
    y_pred = np.zeros(len(dataset_and_labels_per_file))
    y_true = np.zeros(len(dataset_and_labels_per_file))
    for i, (file_features, label) in enumerate(dataset_and_labels_per_file):
        with torch.no_grad():
            features = torch.from_numpy(file_features).to(DEVICE)
            errors = np.mean(
                np.square((file_features - model(features).cpu().numpy())), axis=1
            )
            y_pred[i] = np.mean(errors)
            y_true[i] = label

    return y_pred, y_true


def infer_mlp(model, dataset_and_labels_per_file):
    model.eval()
    model.to(DEVICE)
    dataset_and_labels_per_file = dataset_and_labels_per_file.astype(np.float32)
    features = dataset_and_labels_per_file[:, :-1]
    target = dataset_and_labels_per_file[:, -1]
    y_pred = None
    y_true = target
    with torch.no_grad():
        features = torch.from_numpy(features).to(DEVICE)
        y_pred = model(features).cpu().numpy()
    return y_pred, y_true


def evaluate_model(model, dataset_and_labels_per_file, younden_threshold, infer_func):
    y_pred, y_true = infer_func(model, dataset_and_labels_per_file)
    fpr, tpr, thresholds_roc = metrics.roc_curve(y_true, y_pred, sample_weight=None)
    auc_roc = metrics.auc(fpr, tpr)
    pc, rc, _ = metrics.precision_recall_curve(y_true, y_pred)
    binary_roc_pred = y_pred >= younden_threshold
    cm = metrics.confusion_matrix(y_true, binary_roc_pred)
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / float(tn + fp)
    recall = tp / float(tp + fn)

    y_true = y_true.astype(int)
    binary_roc_pred = binary_roc_pred.astype(int)
    gmean = geometric_mean_score(y_true, binary_roc_pred, average="binary")

    return auc_roc, gmean, tnr, recall


def get_youdens_score(vals_dir):
    scores = []
    for dir in vals_dir:
        file = dir / "youden_score.txt"
        f = open(file)
        score = float(f.readline().replace("\n", ""))
        f.close()
        scores.append(score)
    return scores


@dataclass
class EvalMetrics:
    auc: float
    gmean: float
    recall: float
    tnr: float


@dataclass
class FinalEvaluation:
    exp_name: str
    method: str

    mean_recall: float
    mean_tnr: float

    mean_recall_std: float
    mean_tnr_std: float

    auc_mean: float
    auc_std: float

    gmean_mean: float
    gmean_std: float


def eval_model(test_dir, features, score, category):
    model = torch.load(test_dir / "model" / "best.pt")
    if category == "sound-ae":
        infer_func = infer_ae
    elif category == "sound-mlp":
        infer_func = infer_nn
    else:
        infer_func = infer_mlp
    auc, gmean, tnr, recall = evaluate_model(model, features, score, infer_func)
    return EvalMetrics(auc, gmean, recall, tnr)


def get_final_evaluations(exp_name, exp_dir, category) -> list[FinalEvaluation]:
    val_dir = exp_dir / "val"
    test_dir = exp_dir / "test"
    eval_file = test_dir / "final_eval.fbin"
    if os.path.exists(eval_file):
        f = open(eval_file, "rb")
        return pickle.load(f)

    files = glob.glob(str(val_dir / "**" / "evaluation.txt"), recursive=True)
    numbers = [int(Path(file).parent.name) for file in files]
    groups = utils.get_exps_val_groups_number(numbers)

    group_val_test_paths = []
    for group in groups:
        vals_number = group[0]
        test_number = group[1]
        valspath = [val_dir / f"{i}" for i in vals_number]
        testpath = test_dir / f"{test_number}"
        group_val_test_paths.append((valspath, testpath))

    exp_metrics = {
        "mean_score": [],
        "median_score": [],
        "min_score": [],
        "max_score": [],
    }
    for i, group in enumerate(group_val_test_paths):
        youdens_score = get_youdens_score(group[0])
        youdens_score = np.array(youdens_score)

        mean_score = youdens_score.mean()
        median_score = np.median(youdens_score)
        min_score = youdens_score.min()
        max_score = youdens_score.max()

        # fmt: off
        features = pd.read_csv(group[1] / "eval_dataset.csv").to_numpy()
        saved_file = group[1] / "dataset_and_labels_per_file.bin"
        if os.path.isfile(saved_file):
            with open(saved_file, 'rb') as f:
                dataset_and_labels_per_file = pickle.load(f)
        else:
            if category != "stat-mlp":
                dataset_and_labels_per_file = MelSoundDataset(features).get_dataset_label_per_file()
            else:
                dataset_and_labels_per_file = features
            with open(saved_file, 'wb') as f:
                pickle.dump(dataset_and_labels_per_file, f)
            

        exp_metrics["mean_score"].append(eval_model(group[1], dataset_and_labels_per_file, mean_score, category))
        exp_metrics["median_score"].append(eval_model(group[1], dataset_and_labels_per_file, median_score, category))
        exp_metrics["min_score"].append(eval_model(group[1], dataset_and_labels_per_file, min_score, category))
        exp_metrics["max_score"].append(eval_model(group[1], dataset_and_labels_per_file, max_score, category))

    final_evals = []
    for key, item in exp_metrics.items():
        recalls = np.array([i.recall for i in item])
        tnrs = np.array([i.tnr for i in item])
        aucs = np.array([i.auc for i in item])
        gmeans = np.array([i.gmean for i in item])
        final_eval = FinalEvaluation(
            exp_name=exp_name,
            method=key,
            mean_recall=recalls.mean(),
            mean_recall_std=recalls.std(),
            mean_tnr=tnrs.mean(),
            mean_tnr_std=tnrs.std(),
            auc_mean=aucs.mean(),
            auc_std=aucs.std(),
            gmean_mean=gmeans.mean(),
            gmean_std=gmeans.std(),
        )
        final_evals.append(final_eval)

    with open(eval_file, "wb") as f:
        pickle.dump(final_evals, f)
    return final_evals


def get_exp_dir(exp_number, category):
    if category == "stat-mlp":
        folder = "sound_stat_nn"
    elif category == "sound-ae":
        folder = "mel_sound_ae"
    else:
        folder = "mel_sound_nn"
    return CWD / "experiments" / f"exp_{exp_number}" / folder


def plot_exp(exp_name, exp_numbers, category):
    final_evals = [
        get_final_evaluations(name, get_exp_dir(n, flag), flag)
        for n, flag, name in zip(exp_numbers, category, exp_name)
    ]

    tnrs = []
    recalls = []
    aucs = []
    gmeans = []
    methods = []
    std_tnrs = []
    std_recalls = []
    std_aucs = []
    std_gmeans = []
    names = []
    for n, i in enumerate(final_evals):
        # auc is the same for each threshold
        aucs.append(round(i[0].auc_mean, 2))
        std_aucs.append(round(i[0].auc_std, 2))
        for j in i:
            print(exp_name[n])
            print(j.exp_name)
            assert exp_name[n] == j.exp_name
            names.append(j.exp_name)
            tnrs.append(round(j.mean_tnr, 2))
            recalls.append(round(j.mean_recall, 2))
            methods.append(j.method.replace("_score", "").replace("median", "med"))
            std_tnrs.append(round(j.mean_tnr_std, 2))
            std_recalls.append(round(j.mean_recall_std, 2))
            gmeans.append(round(j.gmean_mean, 2))
            std_gmeans.append(round(j.gmean_std, 2))

    print(names)
    print(methods)
    print(tnrs)
    print(std_tnrs)
    print(recalls)
    print(std_recalls)
    print(aucs)
    print(std_aucs)
    print(gmeans)
    print(std_gmeans)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7})

    width = 0.35  # the width of the bars
    x_pos = np.arange(0, len(tnrs))

    fig, axes = plt.subplots(nrows=3, ncols=1, layout="constrained")
    axes[2].yaxis.grid(True)

    x_pos = x_pos * 0.8
    group_size = len(final_evals[0])
    for i in range(len(final_evals)):
        begin = i * group_size
        end = group_size * (i + 1)
        x_pos[begin:end] = x_pos[begin:end] + i

    tnr_pos = x_pos + width
    recall_pos = tnr_pos + width
    elinewidth = 0.8
    alpha = 0.9
    axes[2].bar(
        tnr_pos,
        tnrs,
        width,
        label="TNR",
        yerr=std_tnrs,
        error_kw={"elinewidth": elinewidth},
        alpha=alpha,
    )
    axes[2].bar(
        recall_pos,
        recalls,
        width,
        label="TPR",
        yerr=std_recalls,
        error_kw={"elinewidth": elinewidth},
        alpha=alpha,
    )

    xticks = recall_pos[1::4] + (width / 2.0)
    minor_ticks = tnr_pos + (width / 2.0)
    axes[2].set_xticks(xticks, exp_name)
    axes[2].set_yticks(np.arange(0, 1 + 0.1, 0.1))
    axes[2].set_xticks(minor_ticks, methods, minor=True)
    axes[2].tick_params(axis="x", which="minor", labelrotation=45.0)

    def set_frescuras(axes):
        # axes.set_ylabel("Score")
        for label in axes.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        axes.set_axisbelow(True)
        # axes.legend(
        #     loc="lower left",
        #     ncols=2,
        #     bbox_to_anchor=(0, 1, 1, 0),
        #     fancybox=True,
        #     edgecolor="black",
        #     fontsize="6",
        # )
        axes.yaxis.grid(True)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.tick_params(bottom=False)

    set_frescuras(axes[2])

    print(methods)
    # plotting gmean
    width = 0.35  # the width of the bars
    x_pos = np.arange(0, len(gmeans))
    x_pos = x_pos * 0.4
    group_size = len(final_evals[0])
    for i in range(len(final_evals)):
        begin = i * group_size
        end = group_size * (i + 1)
        x_pos[begin:end] = x_pos[begin:end] + i * 0.5
    gmean_pos = x_pos + width
    elinewidth = 0.8
    alpha = 0.9
    axes[1].bar(
        gmean_pos,
        gmeans,
        width,
        label="G-means",
        yerr=std_gmeans,
        error_kw={"elinewidth": elinewidth},
        alpha=alpha,
        color="green",
    )
    xticks = gmean_pos[1::4] + (width / 2.0)
    minor_ticks = gmean_pos
    axes[1].set_xticks(xticks, exp_name)
    axes[1].set_yticks(np.arange(0, 1 + 0.1, 0.1))
    axes[1].set_xticks(minor_ticks, methods, minor=True)
    axes[1].tick_params(axis="x", which="minor", labelrotation=45.0)
    set_frescuras(axes[1])

    # plotting aucs
    width = 0.25  # the width of the bars
    x_pos = np.arange(0, len(aucs))
    x_pos = x_pos * 0.37
    aucs_pos = x_pos + width
    elinewidth = 0.8
    alpha = 0.9
    axes[0].bar(
        aucs_pos,
        aucs,
        width,
        label="AUC",
        yerr=std_aucs,
        error_kw={"elinewidth": elinewidth},
        alpha=alpha,
        color="red",
    )
    xticks = aucs_pos
    axes[0].set_yticks(np.arange(0, 1 + 0.1, 0.1))
    axes[0].set_xticks(xticks, map(lambda x: x.replace("\n", ""), exp_name))
    set_frescuras(axes[0])

    fig.legend(
        loc="upper center",
        ncols=4,
        bbox_to_anchor=(0, 1, 1, 0),
        fancybox=True,
        edgecolor="black",
        fontsize="6",
    )
    fig.suptitle(" ")
    # fig.get_constrained_layout()
    fig.set_size_inches(4.65, 4)
    plt.show()


if __name__ == "__main__":
    plot_exp(
        ["\n\nAE", "\n\nMLP", "\n\ncMLP", "\n\nB104", "\n\nB13"],
        [163, 108, 88, 170, 171],
        ["sound-ae", "sound-mlp", "sound-mlp", "stat-mlp", "stat-mlp"],
    )
