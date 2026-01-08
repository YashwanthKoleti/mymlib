import numpy as np
from queue import Queue


class CondiNode:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.is_binary = False
        self.left = None
        self.right = None
        self.value = None
        
    def __repr__(self):
        return (
            f"CondiNode("
            f"feature={self.feature}, "
            f"threshold={self.threshold}, "
            f"is_binary={self.is_binary}, "
            f"value={self.value})"
        )
    
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-9))

def misclass_err(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.max(p)

def impurity(y,kind = 'gini'):
    if kind == 'gini':
        return gini(y)
    if kind == 'entropy':
        return entropy(y)
    if kind == 'misclass_err':
        return misclass_err(y)
    else:
        raise RuntimeError('Choose a valid error/purity type')

class DecisionTree_Classification:
    def __init__(self,max_depth,min_samples_split,min_samples_leaf,threshold,forest = False,criterion = 'gini'):
        self.threshold = threshold
        self.features = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.binary_column = None
        self.classes = None
        self.condition = None
        self.forest = forest
    
    def check(self,X,depth):
        if self.max_depth < depth:
            #print('False1')
            return False
        if self.min_samples_split is not None:
            if X.shape[0] < self.min_samples_split:
                #print('False1')
                return False
        #print('True')
        return True

    def fit(self,X,y):
        if not self.forest:
            self.features = np.arange(X.shape[1])
        else:
            self.features = np.random.randint(0,X.shape[1],size = int(np.sqrt(X.shape[1])))
        self.classes = np.unique(y)
        indices = np.arange(X.shape[0])
    
        self.binary_column = (np.apply_along_axis(
            lambda col: np.unique(col).size == 2,
            axis=0,
            arr=X
        ))

        leafs = []
        condition = CondiNode()
        q = Queue()
        values, counts = np.unique(y, return_counts=True)
        most_repeated = values[np.argmax(counts)]
        condition.value = most_repeated
        q.put((1,condition,indices,most_repeated))
        while not q.empty():
            depth , condition_dummy,indices_dummy,value = q.get()

            if depth >= self.max_depth:
                leafs.append((depth , condition_dummy,indices_dummy,value))
                continue
            if self.check(indices_dummy,depth):
                maxi = 0
                splits = [None,None]
                impurity_parent = impurity(y[indices_dummy],self.criterion)
                for i in self.features:
                    if self.binary_column[i]:
                        indices_left = indices_dummy[X[indices_dummy,i] == 0]
                        indices_right = indices_dummy[X[indices_dummy,i] == 1]

                        if len(indices_left) < self.min_samples_leaf or len(indices_right) < self.min_samples_leaf:
                            continue

                        impurity_left = impurity(y[indices_left],self.criterion)
                        impurity_right = impurity(y[indices_right],self.criterion)
                        impurity_change = impurity_parent - ((len(indices_left)*impurity_left) + (len(indices_right)*impurity_right))/len(indices_dummy)
                        if impurity_change< self.threshold:
                            continue
                        if impurity_change > maxi and depth + 1 <= self.max_depth:
                            condition_dummy.feature = i
                            condition_dummy.threshold = 0.5
                            condition_dummy.is_binary = True

                            maxi = impurity_change
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
                                    impurity_left = impurity(y[indices_left],self.criterion)
                                    impurity_right = impurity(y[indices_right],self.criterion)
                                    impurity_change = impurity_parent - ((len(indices_left)*impurity_left) + (len(indices_right)*impurity_right))/len(indices_dummy)
                                    if impurity_change< self.threshold:
                                        continue
                                    if impurity_change > maxi and depth + 1 <= self.max_depth:
                                            condition_dummy.feature = i
                                            condition_dummy.threshold = mid
                                            condition_dummy.is_binary = False
                                            maxi = impurity_change
                                            splits = [indices_left,indices_right]
                                            
                if splits[0] is not None and splits[1] is not None:
                    condition_dummy.left = CondiNode()
                    condition_dummy.right = CondiNode()

                    values, counts = np.unique(y[indices_left], return_counts=True)
                    most_repeated = values[np.argmax(counts)]
                    condition_dummy.left.value = most_repeated
                    q.put((depth+1, condition_dummy.left, splits[0],most_repeated))

                    values, counts = np.unique(y[indices_right], return_counts=True)
                    most_repeated = values[np.argmax(counts)]
                    q.put((depth+1, condition_dummy.right, splits[1],most_repeated))
                    condition_dummy.right.value = most_repeated
                        
            else:
                leafs.append((condition_dummy,indices_dummy,value))
        self.condition = condition
        return leafs
    
    def _predict(self, x, node=None):
        if node is None:
            node = self.condition  

        if node.left == None:
            return node.value


        if node.is_binary:
            if x[node.feature] == 1:
                return self._predict(x, node.left)
            else:
                return self._predict(x, node.right)


        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

def variance(y):
    return np.var(y)

def mse(y):
    return np.mean((y - np.mean(y))**2)

def mae(y):
    return np.mean(np.abs(y - np.median(y)))

def error(y,kind = 'variance'):
    if kind == 'variance':
        return variance(y)
    if kind == 'mse':
        return mse(y)
    
    if kind == 'mae':
        return mae(y)
    else:
        raise RuntimeError('Choose a valid error/purity type')

class DecisionTree_Regression:
    def __init__(self,max_depth,min_samples_split,min_samples_leaf,threshold,criterion = 'variance'):
        self.threshold = threshold
        self.features = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.binary_column = None
        self.classes = None
        self.condition = None
    
    def check(self,X,depth):
        if self.max_depth < depth:
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
        condition = CondiNode()
        q = Queue()
        condition.value = np.mean(y)
        q.put((1,condition,indices,condition.value))
        while not q.empty():
            depth , condition_dummy,indices_dummy,value = q.get()

            if depth >= self.max_depth:
                leafs.append((depth , condition_dummy,indices_dummy,value))
                continue
            if self.check(indices_dummy,depth):
                maxi = 0
                splits = [None,None]
                error_parent = error(y[indices_dummy],self.criterion)
                for i in range(self.features):
                    if self.binary_column[i]:
                        indices_left = indices_dummy[X[indices_dummy,i] == 0]
                        indices_right = indices_dummy[X[indices_dummy,i] == 1]

                        if len(indices_left) < self.min_samples_leaf or len(indices_right) < self.min_samples_leaf:
                            continue

                        error_left = error(y[indices_left],self.criterion)
                        error_right = error(y[indices_right],self.criterion)
                        error_change = error_parent - ((len(indices_left)*error_left) + (len(indices_right)*error_right))/len(indices_dummy)
                        if error_change< self.threshold:
                            continue
                        if error_change > maxi and depth + 1 <= self.max_depth:
                            condition_dummy.feature = i
                            condition_dummy.threshold = 0.5
                            condition_dummy.is_binary = True

                            maxi = error_change
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
                                    error_left = error(y[indices_left],self.criterion)
                                    error_right = error(y[indices_right],self.criterion)
                                    error_change = error_parent - ((len(indices_left)*error_left) + (len(indices_right)*error_right))/len(indices_dummy)
                                    if error_change< self.threshold:
                                        continue
                                    if error_change > maxi and depth + 1 <= self.max_depth:
                                            condition_dummy.feature = i
                                            condition_dummy.threshold = mid
                                            condition_dummy.is_binary = False
                                            maxi = error_change
                                            splits = [indices_left,indices_right]
                                            
                if splits[0] is not None and splits[1] is not None:
                    condition_dummy.left = CondiNode()
                    condition_dummy.right = CondiNode()


                    condition_dummy.left.value = np.mean(y[splits[0]])
                    condition_dummy.right.value = np.mean(y[splits[1]])

                    q.put((depth+1, condition_dummy.left, splits[0],np.mean(y[indices_left])))
                    q.put((depth+1, condition_dummy.right, splits[1],np.mean(y[indices_right])))
                        
            else:
                leafs.append((condition_dummy,indices_dummy,value))
        self.condition = condition
        return leafs
    
    def _predict(self, x, node=None):
        if node is None:
            node = self.condition  

        if node.left == None:
            return node.value


        if node.is_binary:
            if x[node.feature] == 1:
                return self._predict(x, node.left)
            else:
                return self._predict(x, node.right)


        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])
