# import libraries
import numpy as np
import sklearn
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import os

def preprocess(data):
    x_train = data['train_images'].reshape(-1, 28*28)
    x_val = data['val_images'].reshape(-1, 28*28)
    x_test = data['test_images'].reshape(-1, 28*28)
    y_train = data['train_labels']
    y_val = data['val_labels']
    y_test = data['test_labels']
    return x_train, x_val, x_test, y_train, y_val, y_test
def DecisionTree(x_train, x_test, y_train, y_test):
    if os.path.exists('decisiontree_best_model.pkl'):
        best_model = joblib.load('decisiontree_best_model.pkl')
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("decision_tree score in test:", accuracy)
        return best_model, accuracy
    else:
        param_dist = {
            'criterion': ['gini', 'entropy'],
            'max_depth': np.arange(1, 31),
            'min_samples_split': np.arange(2, 12),
            'min_samples_leaf': np.arange(1, 6)
        }
        model = DecisionTreeClassifier()
        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=25, cv=5, scoring='accuracy',
                                           random_state=42)
        random_search.fit(x_train, y_train)
        validation_score = random_search.score(x_test, y_test)
        print("best_params:", random_search.best_params_)
        print("best_score:", random_search.best_score_)
        print("decision_tree score in test:", validation_score)
        joblib.dump(random_search.best_params_, 'decisiontree_best_params.pkl')
        joblib.dump(random_search.best_estimator_, 'decisiontree_best_model.pkl')
    return random_search, validation_score

data_path_A = r"../Datasets/PneumoniaMNIST/pneumoniamnist.npz"
data_a = np.load(data_path_A)
x_train, x_val, x_test, y_train, y_val, y_test = preprocess(data_a)
x_train = np.vstack((x_train, x_val))
y_train = np.vstack((y_train, y_val))
decisiontree, decisiontree_score = DecisionTree(x_train, x_test, y_train, y_test)

# An example of how the decision tree parameters vary on accuracy
def plot_depth_score():
    scores = []
    for depth in np.arange(1, 31):
        clf = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=5)
        scores.append(np.mean(cv_scores))
    plt.figure()
    plt.plot(np.arange(1, 31), scores, marker='o')
    plt.title('max_depth vs. accuracy')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.grid()
    plt.savefig('max_depth_accuracy.svg', format='svg')
    plt.show()
plot_depth_score()