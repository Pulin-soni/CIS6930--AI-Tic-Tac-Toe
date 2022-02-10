import warnings
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class Classifier():

    def func(self, clf, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=True, test_size=0.20, random_state=42)
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        print("Confusion_matrix for initial training-test set")
        print(confusion_matrix(y_test, y_predicted))
        print("Classification_report for initial training-test set")
        print(classification_report(y_test, y_predicted))
        print("Training score for initial training-test set")
        print(clf.score(X_train, y_train))
        print("Testing score for initial training-test set")
        print(clf.score(X_test, y_test))
        accuracies = cross_val_score(estimator=clf, X=X, y=y, cv=10)
        print("Cross-validation Accuracies:")
        print(accuracies)
        print("Cross-validation Accuracies Mean:")
        print(accuracies.mean())
        print("Cross-validation Accuracies Standard Deviation Mean:")
        print(accuracies.std())
        y_pred_cross_val = cross_val_predict(clf, X, y, cv=10)
        print("Confusion matrix of combined cross-validation data predicted results")
        print(confusion_matrix(y, y_pred_cross_val))
        print("Classification_report of combined cross-validation data predicted results")
        print(classification_report(y, y_pred_cross_val))
        cv_results = cross_validate(clf, X, y, cv=10, return_train_score=True)
        print("Cross-validation testing scores:")
        print(cv_results['test_score'])
        print("Cross-validation training scores")
        print(cv_results['train_score'])


if __name__ == '__main__':
    obj = Classifier()
    data = pd.read_csv('tictac_final.txt', sep=" ", header=None)
    col_X = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    col_y = [9]
    X_final = data[col_X]
    y = data[col_y]

    print("For final boards classification dataset:")
    print("Linear SVM Classifier:")
    clf_svm = svm.SVC(kernel='linear', degree=2, gamma='auto',
                      C=1.2, coef0=0.2, probability=True, random_state=42)

    obj.func(clf_svm, X, y)

    print("MLP Classifier:")
    clf_mlp = MLPClassifier(random_state=42, max_iter=1000,
                            solver='lbfgs', activation='tanh', early_stopping=True)
    obj.func(clf_mlp, X, y)

    print("KNeighborsClassifier:")
    clf_knn = KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='kd_tree',p=2,leaf_size=3,n_jobs=-1)
    obj.func(clf_knn, X, y)

    data_single = pd.read_csv('tictac_single.txt', sep=" ", header=None)

    col_X = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    col_y = [9]
    X_single = data_single[col_X]
    y_single = data_single[col_y]
    
    print("For intermediate boards optimal play(single label) dataset:")
    print("KNeighborsClassifier:")
    clf_knn = KNeighborsClassifier(n_neighbors=9,weights='distance',algorithm='kd_tree',p=2,leaf_size=9,n_jobs=-1)
    obj.func(clf_knn, X_single, y_single)

    print("Linear SVM Classifier:")
    clf_svm = svm.SVC(kernel='linear', degree=9, gamma='auto', C=9, coef0=0.11,
                      probability=True, random_state=42, class_weight='balanced')
    obj.func(clf_svm, X_single, y_single)

    print("MLP Classifier:")
    clf_mlp = MLPClassifier(random_state=42, max_iter=1000,
                            solver='lbfgs', activation='tanh', early_stopping=True)
    obj.func(clf_mlp, X_single, y_single)
