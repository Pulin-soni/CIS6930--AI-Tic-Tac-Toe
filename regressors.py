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
import pickle
from sklearn.metrics import accuracy_score


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class Regressor():

    def funcreg(self,reg, X, y, filename):

        X_train, X_test, y_train, y_test = train_test_split( X, y, shuffle=True, test_size=0.20, random_state=42)
        reg.fit(X_train, y_train)
        y_predicted_proba = reg.predict(X_test)
        weig = 0.5 
        #len(y[y == 1])/len(y[y == 0])
        y_predicted = np.where(y_predicted_proba >= weig, 1, 0)
        print("Confusion_matrix for initial training-test set")
        print(confusion_matrix(y_test, y_predicted))
        print("Classification_report for initial training-test set")
        print(classification_report(y_test, y_predicted))
        #print("Training score for initial training-test set")
        #print(reg.score(X_train, y_train))
        #print("Testing score for initial training-test set")
        #print(reg.score(X_test, y_test))
        accuracies = cross_val_score(estimator=reg, X=X, y=y, cv=10)
        print("Cross-validation-accuracies")
        print(accuracies)
        print("Cross-validation-accuracies Mean:")
        print(accuracies.mean())
        print("Cross-validation-accuracies Standard Devaition:")
        print(accuracies.std())
        pickle.dump(reg, open(filename, 'wb'))
    
    def linear_reg(self, X,y):
        np_x_train, np_x_test, np_y_train, np_y_test = train_test_split( X, y, shuffle=True, test_size=0.20, random_state=42)
        np_x_train=np.array(np_x_train)
        np_y_train=np.array(np_y_train)
        np_x_test=np.array(np_x_test)
        np_y_test=np.array(np_y_test)        
        Y_pred = np.empty((np.shape(np_y_test)[0], np.shape(np_y_test)[1]))
        bias = 1
        for i in range(9):
            y = np_y_train[:, i]
            W = np.linalg.inv(np_x_train.T @ np_x_train) @ np_x_train.T @ y
            W = [weight + bias for weight in W]
            y_pred = np_x_test @ W
            Y_pred[:, i] = y_pred

        Y_pred = (Y_pred == Y_pred.max(axis=1)[:, None]).astype(int)

        total_acc = np.empty(9)
        for i in range(9):
            total_acc[i] = self.get_accuracy_score(np_y_test[:, i],
                                                    Y_pred[:, i], normalized=False)

        acc = np.sum(total_acc) / (np.shape(np_y_test)[0] * 9)
        print("Accuracy LR: {0}".format(acc))

    def get_accuracy_score(self,y_true, y_pred, normalized=True):
        pred_accu = accuracy_score(y_true, y_pred, normalize=normalized)
        return pred_accu

if __name__ == '__main__':
    obj = Regressor()
    data_multiple = pd.read_csv('tictac_multi.txt', sep=" ", header=None)
    col_X = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    col_y = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    X_multiple = data_multiple[col_X]
    y_multiple=data_multiple[col_y]

    print("For intermediate boards optimal play(multi-label) dataset:")
    print("KNN Regressor:")
    for i in col_y:
        filename = 'Model_param_col_'+str(i-9)+'.pkl'
        print("For column no: "+str(i-9))
        reg_knn = KNeighborsRegressor(n_neighbors=9,weights='distance',algorithm='kd_tree',p=2,leaf_size=9,n_jobs=-1)
        obj.funcreg(reg_knn, X_multiple, data_multiple[i],filename)

    print("Linear Regressor:")
    obj.linear_reg(X_multiple,y_multiple)

    print("MLP Regressor:")
    for i in col_y:
        filename = 'Model_param_col_'+str(i-9)+'.pkl'
        print("For column no: "+str(i-9))
        reg_knn = MLPRegressor(random_state=42, max_iter=1000,early_stopping=True,solver='lbfgs', activation='tanh')
        obj.funcreg(reg_knn, X_multiple, data_multiple[i],filename)
