import numpy as np

class PCA():
    def __init__(self,n_componenets):
        self.n_componenets = n_componenets
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self,X):
        self.mean_ = X.mean(axis = 0)
        S = X - self.mean_

        U, S, Vt = np.linalg.svd(S, full_matrices=False)
        self.components_ = Vt.T[:, :self.n_componenets]

        n_samples = X.shape[0]
        eigenvalues = (S ** 2) / (n_samples - 1)

        self.explained_variance_ = eigenvalues[:self.n_componenets]
        self.explained_variance_ratio_ = (
            eigenvalues[:self.n_componenets] / eigenvalues.sum()
        )


    def transform(self,X):
        return (X - self.mean_)@self.components_
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

import numpy as np

def rbf_kernel(X, Y=None, gamma=1.0):
    if Y is None:
        Y = X
    X_norm = np.sum(X**2, axis=1)
    Y_norm = np.sum(Y**2, axis=1)
    dists = X_norm[:, None] + Y_norm[None, :] - 2 * X @ Y.T
    dists = np.maximum(dists, 0)
    return np.exp(-gamma * dists)


def center_kernel(K):
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


class KPCA:
    def __init__(self, n_components, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma

    def fit(self, X):
        self.X_train = X
        self.n = X.shape[0]

        K = rbf_kernel(X, gamma=self.gamma)
        self.K_centered = center_kernel(K)

        eigvals, eigvecs = np.linalg.eigh(self.K_centered)
        idx = np.argsort(eigvals)[::-1]

        self.eigvals = eigvals[idx][:self.n_components]
        self.alphas = eigvecs[:, idx][:, :self.n_components]

        # normalize eigenvectors
        self.alphas /= np.sqrt(self.eigvals * self.n)

        return self

    def transform(self, X):
        K = rbf_kernel(X, self.X_train, gamma=self.gamma)

        # center test kernel
        K_mean_rows = np.mean(K, axis=1, keepdims=True)
        K_mean_train = np.mean(self.K_centered, axis=0)
        K_mean_all = np.mean(self.K_centered)

        Kc = K - K_mean_rows - K_mean_train + K_mean_all

        return Kc @ self.alphas