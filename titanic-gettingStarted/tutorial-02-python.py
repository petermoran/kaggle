"""Getting started with python II and then using a Random Forest.
"""
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
pd.set_option("display.width", 270)


data_dir = "./data"
train_file = os.path.join(data_dir, "train.csv")


if __name__ == "__main__":
    # Preparing the data.
    train = pd.read_csv(train_file)
    train["Maturity"] = train.Age.map(
            lambda age: 1 if age > 18 else 0)
    train["Gender"] = train.Sex.map(
            {'female':0, 'male':1})

    median_age = pd.pivot_table(
            train,
            rows=["Gender", "Pclass"],
            values="Age",
            aggfunc="median")

    idx = [(g, p) for g, p in zip(train.Gender, train.Pclass)]
    train["AgeFill"] = train.Age.copy()

    m = train.Age.isnull().values
    train.loc[m, "AgeFill"] = median_age[idx].values[m]

    train["AgeIsNull"] = train.Age.isnull().astype(int)

    train["FamilySize"] = train.Parch + train.SibSp
    train["Age*Class"] = train.AgeFill*train.Pclass

    data = train.drop(
            ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

    assert data.isnull().sum().sum() == 0

    data = data.values

    # Create the random forest object which will include all the parameters for
    # the fit.
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the training data to the Survived labels and create the decision
    # trees.
    forest = forest.fit(data[:,1:], data[:,0])

    # Take the same decision trees and run it on the test data.
    #output = forest.predict(test_data)

