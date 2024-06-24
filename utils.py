import numpy as np
import glob
import time
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path
from copy import deepcopy
import pickle
from multiprocess import Pool

TQDM_BAR_FORMAT = "{l_bar}{bar:20}{r_bar}"  # tqdm bar format


def get_tqdm(train_loader):
    return tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        bar_format=TQDM_BAR_FORMAT,
    )  # progress bar


def backward_prop(optimizer, loss_func):
    optimizer.zero_grad()
    loss_func.backward()
    optimizer.step()


## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    """Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution."""

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find("Linear") != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        if m.bias is not None:
            m.bias.data.fill_(0)


def find_last_experiment(search_dir):
    folders = glob.glob((search_dir / "*").__str__())
    experiments_number = []
    for folder in folders:
        if "exp_" in folder:
            experiments_number.append(int(folder.split("_")[-1]))

    if len(experiments_number) == 0:
        return 0

    return max(experiments_number)


class LossTrendEarlyStopper:
    def __init__(self, window_size=7, threshold=-0.0001):
        self.losses = [None] * window_size
        self.window_size = window_size
        self.threshold = threshold

    def should_stop(self, loss):
        self.losses.pop(0)
        self.losses.append(loss)
        if None in self.losses:
            return False
        x = np.arange(0, len(self.losses), 1)
        y = np.array(self.losses)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        if m > self.threshold:
            return True

    def reset(self):
        self.losses = [None] * self.window_size


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


def get_exp_evaluation(eval_file):
    f = open(eval_file, "r")
    counter = 0
    data = {}
    for line in f:
        if counter >= 1 and counter <= 4:
            s = line.split(": ")
            assert len(s) == 2
            data.update({s[0]: float(s[1])})
        counter += 1
    return data


def save_stats_from_training(outdir):
    val_stats_dir = outdir / "val"
    test_stats_dir = outdir / "test"

    def save_stats_from_dir(dir):
        evals_files = glob.glob(str(dir / "**" / "evaluation.txt"), recursive=True)
        evals = []
        for file in evals_files:
            evals.append(get_exp_evaluation(file))

        eval_dict = {"auc": [], "acc": [], "tnr": [], "recall": []}
        for eval in evals:
            for key, value in eval.items():
                eval_dict[key].append(value)

        output_dict = {}
        for key, value in eval_dict.items():
            output_dict.update({(key + "_mean"): round(np.array(value).mean(), 3)})
            output_dict.update({(key + "_std"): round(np.array(value).std(), 3)})

        out_file = open(dir / "stats.json", "w")
        json.dump(output_dict, out_file, indent=2)

    save_stats_from_dir(val_stats_dir)
    save_stats_from_dir(test_stats_dir)


class LossComputer: ...


class EpochLossTracker:
    def __init__(
        self, train_loss_computer: LossComputer, val_loss_computer: LossComputer
    ):
        self.train_loss_computer = train_loss_computer
        self.val_loss_computer = val_loss_computer
        self.last_model = None
        self.best_model = None
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def get_best_model(self):
        return self.best_model

    def get_last_model(self):
        return self.last_model

    def get_epochs(self):
        return self.epochs

    def get_train_loss(self):
        return self.train_losses

    def get_val_loss(self):
        return self.val_losses

    def track(self, model, epoch):
        train_loss = self.train_loss_computer.compute_loss(model)
        val_loss = self.val_loss_computer.compute_loss(model)

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if val_loss <= min(self.val_losses):
            self.best_model = model
        self.last_model = model

        self.epochs.append(epoch)


class TimeProfile:
    def __init__(self, outdir):
        self.outdir = outdir
        self.begin = None
        self.end = None

    def start(self):
        self.begin = time.time()

    def stop(self):
        self.end = time.time()

    def time_spent(self):
        return self.end - self.begin


class ExperimentProfiler:
    def __init__(
        self,
        output_dir,
        train_configs,
        time_profiler: TimeProfile,
        epoch_loss_tracker: EpochLossTracker,
    ):
        self.output_dir = output_dir
        self.train_configs = train_configs
        self.time_profiler = time_profiler
        self.epoch_loss_tracker = epoch_loss_tracker

        os.makedirs(self.output_dir, exist_ok=True)

    def _save_timespent(self):
        filename = self.output_dir / "traing_time.txt"
        f = open(filename, "w")
        training_time = self.time_profiler.time_spent() / 60.0
        epochs = self.epoch_loss_tracker.get_epochs()[-1]
        out = f"Training has last: {training_time:.1f} minutes for {epochs+1} epochs"
        f.writelines(out)
        f.close()

    def _save_model(self):
        model_outdir = self.output_dir / "model"
        filename = model_outdir / "best.pt"
        os.makedirs(model_outdir, exist_ok=True)
        torch.save(self.epoch_loss_tracker.get_best_model(), filename)

    def _save_hyperparameters(self):
        filename = self.output_dir / "hyperparameters.json"
        with open(filename, "w") as outfile:
            yaml.dump(self.train_configs, outfile, default_flow_style=False)

    def _save_plot_losses(self):
        epochs = self.epoch_loss_tracker.get_epochs()
        train_loss = self.epoch_loss_tracker.get_train_loss()
        val_loss = self.epoch_loss_tracker.get_val_loss()

        plt.clf()
        plt.plot(epochs, train_loss, "b.-", label="train-loss")
        plt.plot(epochs, val_loss, "r.-", label="val-loss")
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend()
        plt.grid(True)
        filename = self.output_dir / f"train_val_loss.pdf"
        plt.savefig(fname=filename.__str__())

    def _save_loss_data(self):
        filename = self.output_dir / "epoch_train_val_loss.txt"
        # fmt: off
        train_loss = np.array(self.epoch_loss_tracker.get_train_loss()).reshape(-1, 1)
        val_loss = np.array(self.epoch_loss_tracker.get_val_loss()).reshape(-1, 1)
        epoches = np.array(self.epoch_loss_tracker.get_epochs()).reshape(-1, 1)
        # fmt: on
        result = np.concatenate((epoches, train_loss, val_loss), axis=1)
        np.savetxt(
            filename,
            result,
            delimiter=",",
            header="epoche,train_loss,val_loss",
            comments="",
        )

    def save(self):
        self._save_timespent()
        self._save_hyperparameters()
        self._save_plot_losses()
        self._save_model()
        self._save_loss_data()


def get_mean_val_epoch(outdir):
    val_stats_dir = outdir / "val"
    time_files = glob.glob(
        str(val_stats_dir / "**" / "traing_time.txt"), recursive=True
    )
    time_files = [Path(file) for file in time_files]
    sorted_numbers = [int(file.parent.name) for file in time_files]
    sorted_numbers.sort()
    chosen = []
    for number in sorted_numbers:
        chosen.append(number)
        if len(chosen) > 1:
            if chosen[-2] != (number - 1):
                chosen = []

    mean_epoch = 0
    for file in time_files:
        number = int(file.parent.name)
        if number in chosen:
            f = open(file)
            epoch = f.readline().split(" ")[-2]
            epoch = int(epoch)
            mean_epoch += epoch
            f.close()
    mean_epoch = mean_epoch / float(len(chosen))
    # mean_epoch = 30
    return int(mean_epoch)


def concat_subsampling_group(filepath_csv):
    new_groups = ["_2.bortoni", "_3.bortoni"]
    result_table = filepath_csv
    for group in new_groups:
        replace_function = lambda filename: filename.replace("_1.bortoni", group)
        new_split_table = np.copy(filepath_csv)
        new_split_table[:, 0] = np.vectorize(replace_function)(new_split_table[:, 0])
        new_split_table[:, 1] = np.vectorize(replace_function)(new_split_table[:, 1])
        result_table = np.vstack((result_table, new_split_table))
    return result_table


def _get_stack_frame_matrix(filepath):
    bin_file = open(filepath, "rb")
    feature_vector = pickle.load(bin_file).stack_frames_matrix
    bin_file.close()
    return feature_vector


def fetch_features_from_filepath2(filepath_features):
    p = Pool(3)
    start = time.time()
    list_of_audio_files_result = p.map(_get_stack_frame_matrix, filepath_features)
    p.close()
    p.join()
    print(f"time spent: {time.time() - start}")
    return list_of_audio_files_result


def fetch_features_from_filepath(filepath_features):
    def load_dataset_file(filepath_features):
        stack_frame_matrix = _get_stack_frame_matrix(filepath_features[1])
        label_value = 1 if filepath_features[2] >= 1 else 0
        label_stack = np.full(stack_frame_matrix.shape[0], label_value)
        return (stack_frame_matrix, label_stack)

    p = Pool(3)
    start = time.time()
    list_of_audio_files_result = p.map(load_dataset_file, filepath_features)
    p.close()
    p.join()
    print(f"time spent: {time.time() - start}")
    return list_of_audio_files_result


def get_exps_val_groups_number(numbers: list[int]) -> list[tuple[list[int], int]]:
    sorted_numbers = numbers
    sorted_numbers.sort()
    group = []
    numbers_group = []
    test_counter = 0
    for number in sorted_numbers:
        if len(group) > 1:
            if group[-1] != (number - 1):
                test_counter += 1
                numbers_group.append((group, test_counter))
                group = []
        group.append(number)
    numbers_group.append((group, test_counter + 1))

    return numbers_group
