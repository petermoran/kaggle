"""Some basic plots of the data to see what we have.
"""
import os
import pandas as pd
import matplotlib.pyplot as pp
from sklearn.decomposition import PCA


_width = pd.util.terminal.get_terminal_size()[0]
_width = _width - (_width % 10)
pd.set_option("display.width", _width)


data_dir = "./data"
train_file = os.path.join(data_dir, "train.csv")
target_file = os.path.join(data_dir, "trainLabels.csv")


if __name__ == "__main__":
    train = pd.read_csv(train_file, delimiter=',', header=None)
    target = pd.read_csv(target_file, delimiter=',', header=None)

    basic_plots = False
    do_pca = False
    do_isomap = True

    if basic_plots:
        ax = pp.subplot(2, 1, 1)
        train.describe()[1:].plot(legend=False, ax=ax)
        pp.title("Description of training data.")

        ax = pp.subplot(2, 1, 2)
        train.loc[:,:5].plot(legend=False, ax=ax)
        pp.title("First 5 series plotted.")

        pp.show()

    if do_pca:
        x = train.values
        pca = PCA(n_components=3)
        pca.fit(x)
        y = pca.transform(x)
        print 'Orig shape: ', x.shape, 'New shape: ', y.shape

        pp.scatter(y[:,0], y[:,1], c=target.values)
        pp.show()

    if do_isomap:
        x = train.values
        from sklearn.manifold import Isomap
        isomap = Isomap(n_components=2, n_neighbors=20)
        isomap.fit(x)
        y = isomap.transform(x)

        pp.scatter(y[:,0], y[:,1], c=target.values)
        pp.show()
