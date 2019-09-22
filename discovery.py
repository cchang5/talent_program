# talent discovery script
# makes the ML sausage
import features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def prepare_data():
    data = features.make_features().dropna()
    y = data["label"]
    X = data.drop(["label"], axis=1)
    return X, y


def sausage(X, y):
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    model = LogisticRegression()
    model.fit(train_X, train_y)
    prediction = np.array(model.predict_proba(X))
    #for idx in range(len(prediction)):
    #    print(X.index[idx], prediction[idx][0], y[idx])
    result = pd.DataFrame(prediction[:,0], index=X.index, columns = ["predict_proba"])
    result["label"] = y

    print(result.loc["RadiantBeing"])

    #sns.distplot(result["predict_proba"])
    #plt.show()

def main():
    X, y = prepare_data()
    sausage(X, y)


if __name__ == "__main__":
    main()
