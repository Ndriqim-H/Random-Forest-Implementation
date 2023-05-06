import numpy as np
import sys

'''
The nodes of the decision tree. It contains the feature index, threshold, left and right nodes, and the value.
Parameters
    ----------
    feature_index : int
        The index of the feature to be used for splitting the data.
    
    threshold : int
        The threshold value to be used for splitting the data.

    left : Node
        The left node of the decision tree.

    right : Node
        The right node of the decision tree.

    value : int
        The value of the node. It is the class label if the node is a leaf node.
'''
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

'''
The decision tree class. It contains the minimum number of samples to split, the maximum depth of the tree, the number of features to be selected, the feature names, and the class values.
Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    max_depth : int, default=100
        The maximum depth of the tree. If None, then nodes are expanded until
        the recursion limit is nearly reached. Since recursion is used, it's possible for a node not 
        to become pure.

    n_feats : int
        The number of features randomly to be selected features for each tree.

    feature_names : array-like of shape (n_features)
        The feature names of the dataset.

    class_values : array-like of shape (n_classes)
        The class values of the dataset.
'''
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, feature_names=None, class_values=None):
        self.feature_names = feature_names
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.class_values = class_values
        if(max_depth == None):
            self.max_depth = sys.getrecursionlimit() - 100
        
        self.n_feats = n_feats
        self.root = None
    
    '''
    The function to calculate the Gini Impurity of the data.
    Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples)
            The target values (class labels) as integers or strings.

    
    '''
    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    '''
    The function to traverse the tree. It inputs the training data and outputs the root node of the decision tree.

    Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        node : Node
            The root node of the decision tree, or the current node because it uses recursion to reach the result.
    '''

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        feature_val = x[node.feature_index]
        if node.threshold is not None:
            if feature_val <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            return self._traverse_tree(x, node.left)

    '''
    The function to predict the class labels of the data.

    Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

    Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels.
    '''
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    '''
    The function to grow the tree. It uses the _best_criteria function to find the best feature and threshold to split the data. It then uses the _split function to split the data. It then recursively calls itself to grow the tree.
    Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        depth : int
            The current depth of the tree.
    '''

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feat, best_thresh, left, right)
    
    '''
    The function to split the data. It returns the indices of the left and right data.

    Parameters
        ----------
        X_column : array-like of shape (n_samples, n_features)
            The column of the training input samples.

        y : array-like of shape (n_samples,)
            The target values.
            
        feature_index : int
            The index of the feature to be used for splitting the data.

    '''
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh
    '''
    The function to split the data. It returns the indices of the left and right data.

    Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values.

        X_column : array-like of shape (n_samples,)
            The column of the training input samples.

        split_thresh : int or float
            The threshold to split the data.
    '''
    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        nl, nr = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (nl / n) * e_l + (nr / n) * e_r
        ig = parent_entropy - child_entropy
        return ig
    
    '''
    Calculate the entropy of the data.
    '''
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    '''
    Split the data based on the threshold.
    '''
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    

    '''
    Get the leaf value of the node. If the node is empty, randomly select a class.
    '''
    def _leaf_value(self, y):
        if len(y) == 0:
            #Randomly select a class
            return np.random.choice(self.class_values)
        hist = np.bincount(y)
        return np.argmax(hist)


