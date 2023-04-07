from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_tsne(x, y):
    tsne = TSNE(n_components=2)
    x_embedded = tsne.fit_transform(x)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.show()



