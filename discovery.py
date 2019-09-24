# talent discovery script
# makes the ML sausage
import features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import gvar as gv

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot


def prepare_data():
    data = features.make_features().dropna()
    y = data["label"]
    X = data.drop(["label"], axis=1)
    return X, y


def stratcrossvalid(X, y, model, THRESHOLD_LIST=[0.0, 0.5, 1.0], n_splits=5):
    score = []
    skf = StratifiedKFold(n_splits=n_splits)
    for train_idx, test_idx in skf.split(X, y):
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        val_X = X.iloc[test_idx]
        val_y = y.iloc[test_idx]
        model.fit(train_X, train_y)
        score_skf = []
        for THRESHOLD in THRESHOLD_LIST:
            pred_y = np.where(model.predict_proba(val_X)[:, 1] > THRESHOLD, 1, 0)
            score_skf.append(recall_score(val_y, pred_y))
        score.append(score_skf)
    score_mean = [np.mean(s) for s in np.transpose(score)]
    score_sdev = [np.std(s) for s in np.transpose(score)]
    gvscore = [gv.gvar(score_mean[i], score_sdev[i]) for i in range(len(score_mean))]
    return gvscore


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

    probs = model.predict_proba(val_X)[:, 1]
    fpr, tpr, thresholds = roc_curve(val_y, probs)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle="--")
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker=".")
    # show the plot
    pyplot.show()

    # cross validation
    threshlist = np.linspace(0, 1, 21)
    gvscore_list = stratcrossvalid(X, y, model, THRESHOLD_LIST=threshlist)

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


def main():
    X, y = prepare_data()
    sausage(X, y)


if __name__ == "__main__":
    main()
