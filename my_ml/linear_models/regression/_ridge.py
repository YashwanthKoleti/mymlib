import numpy as np
##very intresting method to find intercept
class LinearRegression:
    def __init__(self, lamb = 0.0):
        self.coef_ = None
        self.intercept_ = None
        self.lamb_ = lamb
    
    def fit(self, X, y):
        # 1. Center data
        X_mean = X.mean(axis=0)
        y_mean = y.mean()

        Xc = X - X_mean
        yc = y - y_mean

        # 2. Ridge closed form (NO intercept here)
        d = X.shape[1]
        self.coef_ = np.linalg.solve(
            Xc.T @ Xc + self.lamb_ * np.eye(d),
            Xc.T @ yc
        )

        # 3. Recover intercept
        self.intercept_ = y_mean - X_mean @ self.coef_

        return self
    
    def predict(self,X):
        if self.coef_ == None:
            raise RuntimeError('The model us not fitted yet')
        
        return X @ self.coef_ + self.intercept_