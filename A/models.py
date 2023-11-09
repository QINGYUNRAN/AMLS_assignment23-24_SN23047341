# import libraries
import numpy as np
import sklearn
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import albumentations as A


def augmentation(images):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomCrop(width=20, height=20, p=0.5),
        A.Rotate(limit=45, p=0.9),
        A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120*0.03, p=0.5),
        A.GaussNoise(p=0.5),
        A.Resize(28, 28),
        A.Normalize(),
    ])
    augmented_img = []
    new_images = np.stack([images] * 3, axis=-1)
    for image in new_images:
        transformed = transform(image=image)
        augment_img = transformed["image"]
        augment_img = np.mean(augment_img, axis=-1)
        augmented_img.append(augment_img)
    ll = np.array(augmented_img)
    print(images.shape, ll.shape)
    res = np.concatenate((images, ll), axis=0)
    print(res.shape)
    return res

def preprocess(data):
    x_train = augmentation(data['train_images']).reshape(-1, 28*28)
    x_val = augmentation(data['val_images']).reshape(-1, 28*28)
    x_test = data['test_images'].reshape(-1, 28*28)
    y_train = np.concatenate((data['train_labels'],data['train_labels']), axis=0).reshape(-1,)
    y_val = np.concatenate((data['val_labels'],data['val_labels']), axis=0).reshape(-1,)
    y_test = data['test_labels'].reshape(-1,)
    return x_train, x_val, x_test, y_train, y_val, y_test


def DecisionTree(x_train, x_test, y_train, y_test):
    if os.path.exists('decisiontree_best_model.pkl'):
        best_model = joblib.load('decisiontree_best_model.pkl')
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        decisiontree_params = joblib.load(r"decisiontree_best_params.pkl")
        print("best params on DecisionTree:", decisiontree_params)
        print("decision_tree score in test:", accuracy)
        return best_model, decisiontree_params, accuracy
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
    return random_search, random_search.best_params_, validation_score

def RandomForest(x_train, x_test, y_train, y_test):
    if os.path.exists('randomforest_best_model.pkl'):
        best_model = joblib.load('randomforest_best_model.pkl')
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        randomforest_params = joblib.load(r"randomforest_best_params.pkl")
        print("best params on RandomForest:", randomforest_params)
        print("random forest score in test:", accuracy)
        return best_model, randomforest_params, accuracy
    else:
        param_dist = {
            'n_estimators': np.arange(100, 1000, 50),
            'max_depth': np.arange(1, 50),
            'min_samples_split': np.arange(2, 12),
            'min_samples_leaf': np.arange(1, 6),
            'bootstrap': [True, False]
        }
        model = RandomForestClassifier()

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100,
                                           scoring='accuracy', cv=5, random_state=42, n_jobs=-1)
        random_search.fit(x_train, y_train)

        validation_score = random_search.score(x_test, y_test)
        print("best_params:", random_search.best_params_)
        print("best_score:", random_search.best_score_)
        print("random forest score in test:", validation_score)
        joblib.dump(random_search.best_params_, 'randomforest_best_params.pkl')
        joblib.dump(random_search.best_estimator_, 'randomforest_best_model.pkl')
        return random_search, random_search.best_params_, validation_score

def my_SVC(x_train, x_test, y_train, y_test):
    if os.path.exists('SVC_best_model.pkl'):
        best_model = joblib.load('SVC_best_model.pkl')
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        svc_params = joblib.load(r"SVC_best_params.pkl")
        print("best params on SVC:", svc_params)
        print("SVC score in test:", accuracy)
        return best_model, svc_params, accuracy
    else:
        model = SVC()
        param_dist = {
            'C': np.arange(0.1, 10),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 1],
            'degree': [2, 3, 4],
            'coef0': [0, 1, 2],
            'shrinking': [True, False],
            'class_weight': [None, 'balanced']
        }

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50,
                                           scoring='accuracy', cv=5, random_state=42, n_jobs=-1)

        random_search.fit(x_train, y_train)
        validation_score = random_search.score(x_test, y_test)
        print("best_params:", random_search.best_params_)
        print("best_score:", random_search.best_score_)
        print("SVC score in test:", validation_score)
        joblib.dump(random_search.best_params_, 'SVC_best_params.pkl')
        joblib.dump(random_search.best_estimator_, 'SVC_best_model.pkl')
        return random_search, random_search.best_params_, validation_score


def my_adaboost(x_train, x_test, y_train, y_test):
    if os.path.exists('adaboost_best_model.pkl'):
        best_model = joblib.load('adaboost_best_model.pkl')
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        adaboost_params = joblib.load(r"adaboost_best_params.pkl")
        print("best params on adaboost:", adaboost_params)
        print("adaboost score in test:", accuracy)
        return best_model, svc_params, accuracy
    else:
        model = AdaBoostClassifier()
        param_dist = {
            'n_estimators': np.arange(50, 500, 50),
            'base_estimator': [DecisionTreeClassifier(**decisiontree_params), SVC(**svc_params)],
            'learning_rate': np.random.uniform(0.01, 1.0, 10),
            'algorithm': ['SAMME', 'SAMME.R'],
        }


        random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1,
                                           random_state=42, scoring='accuracy')

        random_search.fit(x_train, y_train)
        validation_score = random_search.score(x_test, y_test)
        print("best_params:", random_search.best_params_)
        print("best_score:", random_search.best_score_)
        print("adaboost score in test:", validation_score)
        joblib.dump(random_search.best_params_, 'adaboost_best_params.pkl')
        joblib.dump(random_search.best_estimator_, 'adaboost_best_model.pkl')
        return random_search, random_search.best_params_, validation_score



data_path_A = r"../Datasets/PneumoniaMNIST/pneumoniamnist.npz"
data_a = np.load(data_path_A)
x_train, x_val, x_test, y_train, y_val, y_test = preprocess(data_a)
x_train = np.vstack((x_train, x_val))
y_train = np.concatenate((y_train, y_val))
decisiontree, decisiontree_params, decisiontree_score = DecisionTree(x_train, x_test, y_train, y_test)
randomforest, randomforest_params, randomforest_score = RandomForest(x_train, x_test, y_train, y_test)
my_svc, svc_params, svc_score = my_SVC(x_train, x_test, y_train, y_test)
# my_xgb, xgb_score = xgboost(x_train, x_test, y_train, y_test)
ada, ada_params, ada_score = my_adaboost(x_train, x_test, y_train, y_test)

# An example of how the decision tree parameters vary on accuracy
def plot_depth_score(x_train, y_train):
    scores = []
    for depth in np.arange(1, 31):
        clf = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=5)
        scores.append(np.mean(cv_scores))
    plt.figure()
    plt.plot(np.arange(1, 31), scores, marker='o')
    plt.title('Effect of max_depth on DecisionTree Accuracy')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.grid()
    plt.savefig('max_depth_accuracy.svg', format='svg')
    plt.show()

def plot_n_forest(x_train, y_train):

    scores = []
    for max_depth in np.arange(1, 31):
        clf = RandomForestClassifier(max_depth=max_depth, random_state=42)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
        scores.append(np.mean(cv_scores))
    plt.figure()
    plt.plot(np.arange(1, 31), scores, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Effect of max_depth on Random Forest Accuracy')
    plt.grid()
    plt.savefig('max_depth_forest.svg', format='svg')
    plt.show()

def plot_SVC(x_train, y_train):

    scores = []
    for c in np.arange(1, 31):
        clf = SVC(C=c*0.1, random_state=42)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
        scores.append(np.mean(cv_scores))
    plt.figure()
    plt.plot(np.arange(1, 31)*0.1, scores, marker='o')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Effect of C on SVC Accuracy')
    plt.grid()
    plt.savefig('C_SVC.svg', format='svg')
    plt.show()

def plot_adaboost(x_train, y_train):

    scores = []
    for n_estimators in np.arange(100, 1000, 50):
        clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
        cv_scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
        scores.append(np.mean(cv_scores))
    plt.figure()
    plt.plot(np.arange(100, 1000, 50), scores, marker='o')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.title('Effect of n_estimators on Adaboost Accuracy')
    plt.grid()
    plt.savefig('adaboost.svg', format='svg')
    plt.show()

# plot_depth_score(x_train, y_train)
# plot_n_forest(x_train, y_train)
# plot_SVC(x_train, y_train)
# plot_xgboost(x_train, y_train)
