# talent discovery script
# makes the ML sausage
import features
import talent_program as tp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from inspect import signature
import gvar as gv

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot


def prepare_data(normalize):
    data = features.make_features(normalize).dropna()
    y = data["label"]
    X = data.drop(["label"], axis=1)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    print(X)
    return X, y


def stratcrossvalid(X, y, THRESHOLD_LIST=[0.0, 0.5, 1.0], n_splits=5):
    score = []
    accuracy = []
    confusion = {thresh: [] for thresh in THRESHOLD_LIST}
    skf = StratifiedKFold(n_splits=n_splits)
    for train_idx, test_idx in skf.split(X, y):
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=1, max_iter=10000
        )
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        val_X = X.iloc[test_idx]
        val_y = y.iloc[test_idx]
        model.fit(train_X, train_y)
        score_skf = []
        accuracy.append(model.score(train_X, train_y))
        for THRESHOLD in THRESHOLD_LIST:
            pred_y = np.where(model.predict_proba(val_X)[:, 1] > THRESHOLD, 1, 0)
            score_skf.append(recall_score(val_y, pred_y))
            confusion[THRESHOLD].append(confusion_matrix(val_y, pred_y))
        score.append(score_skf)
    score_mean = [np.mean(s) for s in np.transpose(score)]
    score_sdev = [np.std(s) for s in np.transpose(score)]
    gvscore = [gv.gvar(score_mean[i], score_sdev[i]) for i in range(len(score_mean))]
    return gvscore, score, model, accuracy, confusion


def sausage(X, y):
    model = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=1, max_iter=10000
    )

    # single test train split
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    model.fit(train_X, train_y)
    prediction = np.array(model.predict_proba(val_X))
    # for idx in range(len(prediction)):
    #    print(X.index[idx], prediction[idx][0], y[idx])
    result = pd.DataFrame(
        prediction[:, 1], index=val_X.index, columns=["predict_proba"]
    )
    result["label"] = val_y

    sorted_recall = np.sort(result[result["label"] == 1]["predict_proba"])
    print("sorted recall (10%):", sorted_recall[int(0.1 * len(sorted_recall))])

    pred_y = np.array(model.predict(val_X))
    print(confusion_matrix(val_y, pred_y))

    print("score:", model.score(train_X, train_y))

    print("coefficients")
    print(model.coef_)

    if True:  # ROC curve
        THRESHOLD_LIST = [0.1]
        probs = model.predict_proba(val_X)[:, 1]
        fpr, tpr, thresholds = roc_curve(val_y, probs)
        # plot no skill
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        for THRESHOLD in THRESHOLD_LIST:
            pred_y = np.where(model.predict_proba(val_X)[:, 1] > THRESHOLD, 1, 0)
            cmatrix = confusion_matrix(val_y, pred_y)
            FPR = cmatrix[0, 1] / (cmatrix[0, 1] + cmatrix[0, 0])
            TPR = cmatrix[1, 1] / (cmatrix[1, 1] + cmatrix[1, 0])
            # ax.axhline(TPR, color="red")
            # ax.axvline(FPR, color="red")
            ax.errorbar(x=[FPR], xerr=[0.05], y=[TPR], yerr=[0.05], color="red")
        ax.errorbar(x=[0, 1], y=[0, 1], linestyle="--", color="k")
        # plot the roc curve for the model
        ax.errorbar(x=fpr, y=tpr, marker=".")
        ax.set_xlabel("False Positive Rate", fontsize=16)
        ax.set_ylabel("True Positive Rate", fontsize=16)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tick_params(axis="both", labelsize=16)
        # show the plot
        plt.savefig(f"./validation/ROC.png", dpi=300, transparent=False)
        # plt.show()

    # cross validation
    threshlist = np.linspace(0, 1, 51)
    gvscore_list, score, model, accuracy, confusion = stratcrossvalid(
        X, y, THRESHOLD_LIST=threshlist
    )
    for key in [0.2]:  # confusion:
        print(key)
        cmatrix = np.array(confusion[key][-1])
        print(cmatrix)
        print("FPR:", cmatrix[0, 1] / (cmatrix[0, 1] + cmatrix[0, 0]))
        print("TPR:", cmatrix[1, 1] / (cmatrix[1, 1] + cmatrix[1, 0]))
    predict_proba = model.predict_proba(X)
    if False:
        for idx, display_name in enumerate(X.index):
            query = f"SELECT id FROM streamer WHERE display_name='{display_name}';"
            postgres = tp.Postgres()
            streamer_id = np.array(postgres.rawselect(query))[0, 0]
            postgres.close()
            proba = predict_proba[idx][1]
            query = f"INSERT INTO prediction (streamer_id, proba) VALUES({streamer_id}, {proba});"
            postgres = tp.Postgres()
            postgres.rawsql(query)
            postgres.close()

    # x-valid results
    print("Threshold:", threshlist)
    print("Xvalid recall:", gvscore_list)
    print("Xvalid accuracy:", np.mean(accuracy), np.std(accuracy, ddof=1))
    print("coeff:", model.coef_)

    if True:
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        ax.errorbar(
            x=threshlist,
            y=[i.mean for i in gvscore_list],
            yerr=[i.sdev for i in gvscore_list],
        )
        plt.draw()
        plt.show()
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="recall")
    print(scores)
    print("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(scores)
    print("f1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    """

    # sns.distplot(result["predict_proba"])
    # plt.show()

    # Precision recall
    fig = plt.figure("precision recall", figsize=(7, 7))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    precision, recall, _ = precision_recall_curve(val_y, probs)
    step_kwargs = (
        {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
    )
    ax.step(recall, precision, color="#6441A4", where="post")
    ax.fill_between(recall, precision, color="#6441A4", **step_kwargs)

    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(f"./validation/precision_recall.png", dpi=300, transparent=False)
    plt.show()


def svm_sausage(X, y):
    from sklearn.svm import SVC

    model = SVC(gamma="scale", kernel="linear", probability=True)
    skf = StratifiedKFold(n_splits=5)
    pred_y = []
    score = []
    recall = []
    for train_idx, test_idx in skf.split(X, y):
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        val_X = X.iloc[test_idx]
        val_y = y.iloc[test_idx]
        model.fit(train_X, train_y)
        pred_ysplit = model.predict(val_X)
        pred_y.append(pred_ysplit)
        score.append(model.score(val_X, val_y))
        recall.append(recall_score(val_y, pred_ysplit))
    print(score)
    print(np.mean(score), np.std(score))
    print(recall)
    print(np.mean(recall), np.std(recall))

    pred_proba = model.predict_proba(val_X)[:, 1]
    fpr, tpr, thresholds = roc_curve(val_y, pred_proba)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    cmatrix = confusion_matrix(val_y, pred_y[-1])
    FPR = cmatrix[0, 1] / (cmatrix[0, 1] + cmatrix[0, 0])
    TPR = cmatrix[1, 1] / (cmatrix[1, 1] + cmatrix[1, 0])
    print(cmatrix)
    ax.errorbar(x=[0, 1], y=[0, 1], linestyle="--", color="k")
    # plot the roc curve for the model
    ax.errorbar(x=fpr, y=tpr, marker=".")
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tick_params(axis="both", labelsize=16)
    # show the plot
    plt.savefig(f"./validation/svm_ROC.png", dpi=300, transparent=False)
    # plt.show()

    # Precision recall
    fig = plt.figure("precision recall", figsize=(7, 7))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    precision, recall, _ = precision_recall_curve(val_y, pred_proba)
    step_kwargs = (
        {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
    )
    ax.step(recall, precision, color="#6441A4", where="post")
    ax.fill_between(recall, precision, color="#6441A4", **step_kwargs)

    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(f"./validation/svm_precision_recall.png", dpi=300, transparent=False)
    plt.show()


def knn_sausage(X, y):
    from sklearn.neighbors import KNeighborsClassifier

    result = dict()
    auclist = []
    nmin = 7
    nmax = 8
    fig = plt.figure("precision recall", figsize=(7, 7))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    for neigh in range(nmin, nmax):
        model = KNeighborsClassifier(n_neighbors=neigh)
        skf = StratifiedKFold(n_splits=5)
        pred_y = []
        score = []
        recall = []
        for train_idx, test_idx in skf.split(X, y):
            train_X = X.iloc[train_idx]
            train_y = y.iloc[train_idx]
            val_X = X.iloc[test_idx]
            val_y = y.iloc[test_idx]
            model.fit(train_X, train_y)
            pred_ysplit = model.predict(val_X)
            pred_y.append(pred_ysplit)
            score.append(model.score(val_X, val_y))
            recall.append(recall_score(val_y, pred_ysplit))
        print(score)
        print(np.mean(score), np.std(score))
        print(recall)
        print(np.mean(recall), np.std(recall))
        result[neigh] = [
            gv.gvar(np.mean(score), np.std(score)),
            gv.gvar(np.mean(recall), np.std(recall)),
        ]
        pred_proba = model.predict_proba(val_X)[:, 1]
        precision, recall, _ = precision_recall_curve(val_y, pred_proba)
        step_kwargs = (
            {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
        )
        ax.step(recall, precision, where="post", label=neigh)
        # ax.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        auclist.append(auc(recall, precision))

    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.show()  # END precisioin recall

    # plot AUC for precisioin recall
    fig = plt.figure("precision recall auc", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(x=np.arange(nmin, nmax), y=auclist)
    ax.set_xlabel("Nearest Neighbors", fontsize=16)
    ax.set_ylabel("Precision-Recall AUC", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.savefig(f"./validation/knn_auc.png", dpi=300, transparent=False)
    plt.show()

    fig = plt.figure("recall", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(
        x=np.arange(nmin, nmax),
        y=[result[idx][1].mean for idx in np.arange(nmin, nmax)],
        yerr=[result[idx][1].sdev for idx in np.arange(nmin, nmax)],
        label="recall",
        color="#6441A4",
    )
    ax.set_xlabel("Nearest Neighbors", fontsize=16)
    ax.set_ylabel("Recall", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.savefig(f"./validation/knn_neigh.png", dpi=300, transparent=False)
    fig = plt.figure("accuracy", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(
        x=np.arange(nmin, nmax),
        y=[result[idx][0].mean for idx in np.arange(nmin, nmax)],
        yerr=[result[idx][0].sdev for idx in np.arange(nmin, nmax)],
        label="accuracy",
    )
    plt.show()

    pred_proba = model.predict_proba(val_X)[:, 1]
    fpr, tpr, thresholds = roc_curve(val_y, pred_proba)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    cmatrix = confusion_matrix(val_y, pred_y[-1])
    FPR = cmatrix[0, 1] / (cmatrix[0, 1] + cmatrix[0, 0])
    TPR = cmatrix[1, 1] / (cmatrix[1, 1] + cmatrix[1, 0])
    print(cmatrix)
    ax.errorbar(x=[0, 1], y=[0, 1], linestyle="--", color="k")
    # plot the roc curve for the model
    ax.errorbar(x=fpr, y=tpr, marker=".")
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tick_params(axis="both", labelsize=16)
    # show the plot
    plt.savefig(f"./validation/knn_ROC.png", dpi=300, transparent=False)
    # plt.show()

    # Precision recall
    fig = plt.figure("precision recall", figsize=(7, 7))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    precision, recall, _ = precision_recall_curve(val_y, pred_proba)
    step_kwargs = (
        {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
    )
    ax.step(recall, precision, color="#6441A4", where="post")
    ax.fill_between(recall, precision, color="#6441A4", **step_kwargs)

    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(f"./validation/knn_precision_recall.png", dpi=300, transparent=False)
    plt.show()


def main():
    X, y = prepare_data(normalize=True)
    sausage(X, y)
    svm_sausage(X, y)
    knn_sausage(X, y)


if __name__ == "__main__":
    main()
