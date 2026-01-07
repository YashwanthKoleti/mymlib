import numpy as np
from queue import Queue

def gini(y,classes):
    ans = 1
    for c in classes:
        ans = ans - ((np.sum(y == c))/len(y))**2

    return ans

class DecisionTree:
    def __init__(self,max_depth,min_samples_split,min_samples_leaf,threshold,criterion = 'gini'):
        self.threshold = threshold
        self.features = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.binary_column = None
        self.classes = None
    
    def check(self,X):
        if self.max_depth <= 0:
            #print('False1')
            return False
        if self.min_samples_split is not None:
            if X.shape[0] < self.min_samples_split:
                #print('False1')
                return False
        #print('True')
        return True

    def fit(self,X,y):
        self.features = X.shape[1]
        self.classes = np.unique(y)
        indices = np.arange(X.shape[0])
    
        self.binary_column = (np.apply_along_axis(
            lambda col: np.unique(col).size == 2,
            axis=0,
            arr=X
        ))

        leafs = []
        q = Queue()
        q.put(indices)
        while not q.empty():
            indices_dummy = q.get()
            if self.check(indices_dummy):
                maxi = 0
                splits = [None,None]
                gi = gini(y[indices_dummy],self.classes)
                for i in range(self.features):
                    if self.binary_column[i]:
                        indices_left = indices_dummy[X[indices_dummy,i] == 1]
                        indices_right = indices_dummy[X[indices_dummy,i] == 0]

                        if len(indices_left) < self.min_samples_leaf or len(indices_right) < self.min_samples_leaf:
                            continue

                        gi_left = gini(y[indices_left],self.classes)
                        gi_right = gini(y[indices_right],self.classes)
                        gi_increase = gi - ((len(indices_left)*gi_left) + (len(indices_right)*gi_right))/len(indices)
                        if gi_increase< self.threshold:
                            continue
                        if gi_increase > maxi :
                            maxi = gi_increase
                            splits = [indices_left,indices_right]
                    else:
                            thresholds_dummy = np.sort(np.unique(X[indices_dummy, i]))
                            for k in range(1,len(thresholds_dummy)):
                                mid = (thresholds_dummy[k] + thresholds_dummy[k-1])/2
                                indices_left = indices_dummy[X[indices_dummy,i] <= mid]
                                indices_right = indices_dummy[X[indices_dummy,i] > mid]

                                if len(indices_left) < self.min_samples_leaf or len(indices_right) < self.min_samples_leaf:
                                    continue
                                else:
                                    gi_left = gini(y[indices_left],self.classes)
                                    gi_right = gini(y[indices_right],self.classes)
                                    gi_increase = gi - ((len(indices_left)*gi_left) + (len(indices_right)*gi_right))/len(indices_dummy)
                                    if gi_increase< self.threshold:
                                        continue
                                    if gi_increase > maxi :
                                            maxi = gi_increase
                                            splits = [indices_left,indices_right]
                if splits[0] is not None and splits[1] is not None:
                    leafs.append(splits[0])
                    leafs.append(splits[1])
                        
            else:
                leafs.append(indices_dummy)
        
        return leafs