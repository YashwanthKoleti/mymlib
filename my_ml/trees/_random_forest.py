import numpy as np
from ._decision_tree import DecisionTree_Regression ,DecisionTree_Classification

class RandomForestClassification:
    def __init__(self,threshold,min_samples_leaf,num_trees = 20,criterion='gini', max_depth=5, min_samples_split=2):
        self.num_trees = num_trees
        self.trees = [
    DecisionTree_Classification(
        threshold=threshold,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        forest=True
    )
    for _ in range(self.num_trees)
]

        self.threshold = threshold
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_depth = max_depth
        self.sample_size = None

    def fit(self,X,y):
        self.sample_size = X.shape[0]
        for tree in self.trees:
            indices = np.random.choice(self.sample_size, size=self.sample_size, replace=True)
            tree.fit(X[indices],y[indices])
    
    def _predict(self, x):
        preds = np.array([
        tree.predict(x.reshape(1, -1))[0]
        for tree in self.trees
        ])
        values, counts = np.unique(preds, return_counts=True)
        return values[np.argmax(counts)]


    def predict(self,X):
        return np.array([self._predict(x) for x in X])