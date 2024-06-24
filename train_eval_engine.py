import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import xgboost as xgb
from sklearn.svm import SVC


def binarize_array(arr_multiclass):
    arr = arr_multiclass
    filter_ = arr >= 1
    arr[filter_] = 1
    return arr


def eval(model, eval_features, eval_target):
    pred = model.predict(eval_features)
    tn, fp, fn, tp = metrics.confusion_matrix(eval_target, pred).ravel()
    tnr = tn / float(tn + fp)
    recall = tp / float(tp + fn)
    return tnr, recall


def get_eval_string(tnr_list, recall_list):
    tnr_np = np.array(tnr_list)
    tnr_mean = tnr_np.mean()
    tnr_std = tnr_np.std()

    recall_np = np.array(recall_list)
    recall_mean = recall_np.mean()
    recall_std = recall_np.std()

    s = f"tnr_mean: {tnr_mean}\ntnr_std: {tnr_std}\nrecall_mean: {recall_mean}\nrecall_std: {recall_std}\n"
    return s


def has_method(o, name):
    return callable(getattr(o, name, None))


def train_eval(model_parameters, model_class, features, target, eval_function=None):
    if eval_function is None:
        eval_function = eval
    test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    val_tnr_list = []
    val_recall_list = []
    test_tnr_list = []
    test_recall_list = []
    for train_index, test_index in test_skf.split(features, target):

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
        y_train_binary = binarize_array(y_train)
        y_test_binary = binarize_array(y_test)

        val_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for cv_train_index, cv_val_index in val_skf.split(X_train, y_train):
            training_features, val_features = (
                X_train[cv_train_index],
                X_train[cv_val_index],
            )
            training_target, val_target = (
                y_train_binary[cv_train_index],
                y_train_binary[cv_val_index],
            )

            model = model_class(**model_parameters)
            if has_method(model, "set_validation"):
                model.set_validation(val_features, val_target)
            model.fit(training_features, training_target)

            tnr, recall = eval_function(model, val_features, val_target)
            val_tnr_list.append(tnr)
            val_recall_list.append(recall)

        model = model_class(**model_parameters)
        model.fit(X_train, y_train_binary)
        tnr, recall = eval_function(model, X_test, y_test_binary)
        test_tnr_list.append(tnr)
        test_recall_list.append(recall)

    val_eval_string = get_eval_string(val_tnr_list, val_recall_list)
    print("Val: \n")
    print(val_eval_string)

    print("Test:\n")
    test_eval_string = get_eval_string(test_tnr_list, test_recall_list)
    print(test_eval_string)
