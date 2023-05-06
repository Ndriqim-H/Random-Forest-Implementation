from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import DecisionTreeImplementation as dti

'''
The model training function, it inputs the training data and outputs a list of decision trees.
Parameters
    ----------
    X_Train : array-like of shape (n_samples, n_features)
        The training input samples.

    X_Test : array-like of shape (n_samples, n_features)
        The training input samples.

    n_features : int
        The number of features randomly to be selected features for each tree.

    n_estimators : int, default=100
        The number of trees in the forest.

    n_samples : int
        The number of samples to be selected during the "Bootstrapping" process.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

Returns
-------
    trees : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
'''

def Random_Forest_Fit(X_Train, X_Test, n_features, n_samples, n_estimators=100, max_depth=None, class_values=None):
    trees = []
    for i in range(n_estimators):
        sample = resample(X_Train, X_Test, replace=True, n_samples=n_samples)
        X_train = sample[0]
        Y_train = sample[1]
        
        features = resample(X_train.columns, replace=False, n_samples=n_features)
        tree = dti.DecisionTree(max_depth=max_depth, n_feats=n_features, feature_names=features, class_values= class_values)
        X_train = X_train[features]
        X = X_train.values
        y = Y_train.values
        tree.fit(X, y)
        trees.append(tree)
    return trees

def Random_Forest_Predict(X, trees):
    predictions = np.zeros((len(X), len(trees)))
    for i, tree in enumerate(trees):
        features = tree.feature_names
        X_test = X[features].values
        predictions[:, i] = tree.predict(X_test)
    final_predictions = []
    for i in range(len(X)):
        counts = np.bincount(predictions[i].astype(int))
        final_predictions.append(np.argmax(counts))
    return np.array(final_predictions)


from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('heart.csv')
class_name = 'output'
X = df.drop(class_name, axis=1)
y = df[class_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = []

trees = Random_Forest_Fit(X_train, y_train, 5, 30, 100, None, class_values=df[class_name].unique())
y_pred = Random_Forest_Predict(X_test, trees)


accuracy = accuracy_score(y_test, y_pred)
print('My implementation of RF Accuracy:', accuracy)


# Now we test the accuracy of the Random Forest Classifier from sklearn to see how well our implementation did.

from sklearn.ensemble import RandomForestClassifier
# Create an instance of the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None)

# Fit the Random Forest Classifier to the training data
rf.fit(X_train, y_train)
y_pred1 = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred1)
print('SKLearn RF Accuracy:', accuracy)




