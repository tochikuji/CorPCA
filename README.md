## CorPCA

### Description

The implementation of principal component analysis with the mean-squared error (MSE) minimization criterion; a.k.a. Karhunen-Loeve expantion.

The class `CorPCA` has a same interface as `sklearn.decomposition.PCA`, which has `fit`, `transform`, `fit_transform` inherit from `sklearm.base.TransformMixin`.

This package also provides a generalized PCA class `corpca.PCA` which takes a parameter `criterion` to switch the type of Gram matrix, e.g.,

```
from corpca import PCA

PCA(criterion='variance', n_comonents=10)
# => sklearn.decomposition.PCA(n_components=10)

PCA(criterion='mse', n_components=10)
# => corpca.CorPCA(n_components=10)
```

### Installation

`pip install .`

or from PyPI

`pip install corpca`

### LICENSE

MIT

### Author

Aiga SUZUKI [tochikuji@gmail.com]
