"""Doing the first excel tutorial using pandas.
"""
import pandas as pd
import os
pd.set_option("display.width", 270)


data_dir = "./data"
train_file = os.path.join(data_dir, "train.csv")


if __name__ == "__main__":
    train = pd.read_csv(train_file)
    train["Maturity"] = train.Age.map(
            lambda age: "adult" if age > 18 else "child")

    # Using pivot.
    pvt = train.pivot(index="Sex", columns="PassengerId", values="Survived")

    print "Percentage of sex that survives:"
    print pvt.mean(axis=1)

    # Using pivot_table.
    tbl = pd.pivot_table(
            train,
            rows=["Sex", "Maturity"],
            values="Survived",
            aggfunc="mean")
    print "Using pivot_table:"
    print tbl

    # Adding passenger class.
    tbl = pd.pivot_table(
            train,
            rows=["Sex", "Pclass"],
            values="Survived",
            aggfunc="mean")

    print "Adding passenger class\n", tbl

    # Fare paid.
    i = pd.cut(train.Fare, [-1, 10, 20, 30, train.Fare.max()], labels=False)
    train["FareBin"] = i

    # Adding passenger class.
    tbl = pd.pivot_table(
            train,
            rows=["Sex", "Pclass", "FareBin"],
            values="Survived",
            aggfunc="mean")

    print "Adding fare bin\n", tbl

