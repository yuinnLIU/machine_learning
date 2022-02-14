##用sklearn的PCA
from sklearn.decomposition import PCA
import numpy as np
# x = np.array([[-1, -1, 0, 2, 0],
#               [-2, 0, 0, 1, 1]])

X = np.array([[-1, -2], [-1, 0], [0, 0], [2, 1], [0, 1]])
pca=PCA(n_components=1)
pca.fit(X)
print(pca.transform(X))