import torch
import numpy as np
from pathlib import Path
import os
import yaml
from train_eval_engine import train_eval
from torch_models import NNClassifier
from torch import nn
import utils
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import metrics

CWD = Path(os.getcwd())
CONFIG_FILE = "./config.yaml"
DEVICE = 0  # gpu device


def build_nn(n_input, activation_function, n_innerlayers, n_per_layer, bias):
    nn = []
    input = [
        ["BatchNorm1d", [n_input]],
        ["Linear", [n_input, n_per_layer, bias]],
        [activation_function, [True]],
    ]
    innerlayers = [
        ["Linear", [n_per_layer, n_per_layer, bias]],
        [activation_function, [True]],
    ]
    output = [["Linear", [n_per_layer, 1, bias]], ["Sigmoid", []]]

    nn.append(input)
    for i in range(n_innerlayers):
        nn.append(innerlayers)
    nn.append(output)
    return nn


def get_training_objects(configs):
    epochs = configs["train_config"]["epochs"]
    model = NNClassifier(configs["train_config"]["model"]["NNClassifier"]).to(DEVICE)
    print(model)
    model.apply(utils.weights_init_normal)
    model.train()
    loss_function = nn.BCELoss()
    lr = configs["train_config"]["lr"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=configs["train_config"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(float(configs["train_config"]["StepLR"]["step_size"]) * epochs),
        gamma=float(configs["train_config"]["StepLR"]["gamma"]),
    )
    return model, loss_function, optimizer, epochs, scheduler


class StatDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        label = self.target[idx]
        return torch.from_numpy(feature_vector).to(DEVICE), torch.tensor(
            label, dtype=torch.float32
        ).to(DEVICE)


def _get_loader(features, target, batch_size):
    dataset = StatDataset(features, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


class LossComputer:
    def __init__(self, loss_function, features, targets):
        self.loss_function = loss_function
        self.features = torch.from_numpy(features).to(DEVICE)
        self.targets = torch.from_numpy(targets).to(DEVICE)
        self.loss = None

    def loss_value(self):
        return self.loss

    def compute_loss(self, model):
        model.eval()
        with torch.no_grad():
            output = model(self.features)
            loss = self.loss_function(output.flatten(), self.targets)
            self.loss = loss.item()
        return self.loss


class StatNNClassifier:
    INSTANCES_COUNTER = 0
    VAL_INSTANCES_COUNTER = 0
    TEST_INSTANCES_COUNTER = 0

    def __init__(self, train_config, outdir):
        StatNNClassifier.INSTANCES_COUNTER += 1
        self.train_config = train_config
        (
            self.model,
            self.loss_function,
            self.optmizer,
            self.total_epochs,
            self.scheduler,
        ) = get_training_objects(self.train_config)
        self.early_stopper = utils.LossTrendEarlyStopper(20)
        self.epoch_loss_tracker: utils.EpochLossTracker = None
        self.outdir = outdir

        self.val_feature = None
        self.val_target = None

    def set_validation(self, val_feature, val_target):
        self.val_feature = val_feature
        self.val_target = val_target

    def best_model(self):
        return self.epoch_loss_tracker.get_best_model()

    def fit(self, training_features, training_target):

        train_loss_computer = LossComputer(
            self.loss_function, training_features, training_target
        )
        if self.val_feature is None:
            StatNNClassifier.TEST_INSTANCES_COUNTER += 1
            val_loss_computer = train_loss_computer
            self.total_epochs = utils.get_mean_val_epoch(self.outdir)
            self.outdir = (
                self.outdir / "test" / str(StatNNClassifier.TEST_INSTANCES_COUNTER)
            )
        else:
            StatNNClassifier.VAL_INSTANCES_COUNTER += 1
            self.outdir = self.outdir / "val" / str(StatNNClassifier.INSTANCES_COUNTER)
            val_loss_computer = LossComputer(
                self.loss_function, self.val_feature, self.val_target
            )

        self.epoch_loss_tracker = utils.EpochLossTracker(
            train_loss_computer, val_loss_computer
        )
        time_profiler = utils.TimeProfile(self.outdir)
        experiment_profiler = utils.ExperimentProfiler(
            self.outdir, self.train_config, time_profiler, self.epoch_loss_tracker
        )

        train_loader = _get_loader(
            training_features,
            training_target,
            self.train_config["train_config"]["batch_size"],
        )

        time_profiler.start()
        for epoch in range(self.total_epochs):
            self.model.train()
            mloss = 0
            pbar = utils.get_tqdm(train_loader)
            for batch_idx, train_set in pbar:
                # ===================forward=====================
                feature_vectors, labels = train_set
                output = self.model(feature_vectors)
                loss = self.loss_function(output.flatten(), labels)
                # ===================backward====================
                utils.backward_prop(self.optmizer, loss)
                # ===================Stats=======================
                mloss += loss.item()
                pbar.set_description(
                    f"{epoch}/{self.total_epochs - 1} mloss: {mloss/(batch_idx + 1):.6f}"
                )

            self.scheduler.step()
            self.epoch_loss_tracker.track(self.model, epoch)
            if self.early_stopper.should_stop(val_loss_computer.loss_value()):
                break

        time_profiler.stop()
        experiment_profiler.save()


def evaluate_model(model, features, target):
    model.eval()
    y_pred = None
    y_true = target
    with torch.no_grad():
        features = torch.from_numpy(features).to(DEVICE)
        y_pred = model(features).cpu().numpy()

    fpr, tpr, thresholds_roc = metrics.roc_curve(y_true, y_pred, sample_weight=None)
    auc_roc = metrics.auc(fpr, tpr)
    idx = np.argmax(tpr - fpr)
    cut_off_threshold_roc = thresholds_roc[idx]
    binary_roc_pred = y_pred >= cut_off_threshold_roc
    acc_roc = metrics.accuracy_score(y_true, binary_roc_pred)
    cm_roc = metrics.confusion_matrix(y_true, binary_roc_pred)
    tn, fp, fn, tp = cm_roc.ravel()
    tnr = tn / float(tn + fp)
    recall = tp / float(tp + fn)

    return auc_roc, acc_roc, cm_roc, tnr, recall, cut_off_threshold_roc


def evaluation_function(model, features, target):
    def save_eval_features(features, outdir):
        pd.DataFrame(features).to_csv(
            outdir / "eval_dataset.csv",
            sep=",",
            index=False,
        )

    output_dir = model.outdir
    save_eval_features(
        np.concatenate((features, target.reshape(-1, 1)), axis=1), output_dir
    )
    auc, acc, cm, tnr, recall, youden_score = evaluate_model(
        model.best_model(), features, target
    )

    def save_eval():
        filename = output_dir / "evaluation.txt"
        f = open(filename, "w")
        out = f"Val:\nauc: {auc:.3f}\nacc: {acc:.3f}\ntnr: {tnr}\nrecall: {recall}\n"
        out += utils.print_cm(cm)
        f.writelines(out)
        f.close()

    def save_youden_score():
        filename = output_dir / "youden_score.txt"
        f = open(filename, "w")
        out = f"{youden_score}"
        f.writelines(out)
        f.close()

    save_eval()
    save_youden_score()
    return tnr, recall


def get_all_stats_features():
    dataset = pd.read_csv("stat_dataset.csv", header=None).to_numpy()
    features = dataset[:, :-1].astype(np.float32)
    target = dataset[:, -1].astype(np.float32)
    n_input = 104
    return features, target, n_input


def get_sound_stats_features():
    dataset = pd.read_csv("stat_dataset.csv", header=None).to_numpy()
    features = dataset[:, -14:-1].astype(np.float32)
    target = dataset[:, -1].astype(np.float32)
    n_input = 13
    return features, target, n_input


def train_folds_grid_search(configs):
    # features, target, n_input = get_all_stats_features()
    features, target, n_input = get_sound_stats_features()

    lrs = [5 * 1e-4]
    innerlayers = [1]
    wds = [1e-5]
    n_per_layer = [n_input]
    activation_function = ["ReLU"]
    bias = [True]
    for b in bias:
        for lr in lrs:
            for innerlayer in innerlayers:
                for neuron_per_layer in n_per_layer:
                    for wd in wds:
                        for af in activation_function:
                            configs["train_config"]["weight_decay"] = wd
                            configs["train_config"]["lr"] = lr
                            configs["train_config"]["model"]["NNClassifier"] = build_nn(
                                n_input=n_input,
                                activation_function=af,
                                n_innerlayers=innerlayer,
                                n_per_layer=neuron_per_layer,
                                bias=b,
                            )

                            last_exp = utils.find_last_experiment(CWD / "experiments")
                            outdir = (
                                CWD
                                / "experiments"
                                / f"exp_{last_exp + 1}"
                                / "sound_stat_nn"
                            )
                            model_class = StatNNClassifier
                            model_parameters = {
                                "train_config": configs,
                                "outdir": outdir,
                            }
                            train_eval(
                                model_parameters,
                                model_class,
                                features,
                                target,
                                eval_function=evaluation_function,
                            )
                            StatNNClassifier.INSTANCES_COUNTER = 0
                            StatNNClassifier.VAL_INSTANCES_COUNTER = 0
                            StatNNClassifier.TEST_INSTANCES_COUNTER = 0
                            utils.save_stats_from_training(outdir)


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    with open(CONFIG_FILE) as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)

    train_folds_grid_search(configs)
