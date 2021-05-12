import numpy as np
from glob import glob
import os
import cv2

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import svm
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import tensorflow as tf

class0_path = "/Users/xinxinyang/data/normal_cut_1/"
output0_path = "/Users/xinxinyang/data1/normal/"

class1_path = "/Users/xinxinyang/data/tumor_cut_1/"
output1_path = "/Users/xinxinyang/data1/tumor/"


def save_image(input, output):
    file_list=glob(input+"*.png")
    for fcount, img_file in enumerate(file_list):
        try:
            image = cv2.imread(img_file)
        except:
            print("can't read file")
        image = cv2.resize(image, (128, 128), cv2.INTER_LINEAR)
        np.save(os.path.join(output,"images_%04d.npy" % fcount),image)

# save_image(class0_path, output0_path)
# save_image(class1_path, output1_path)

def save_file():
    X = []
    Y = []
    for img_file in glob(output0_path + '*.npy'):
            img = np.load(img_file).astype(np.float32)
            X.append(img)
            Y.append(0)
    for img_file in glob(output1_path + '*.npy'):
            img = np.load(img_file).astype(np.float32)
            X.append(img)
            Y.append(1)

    X = np.reshape(X,(-1,128,128,3))

    np.save("/Users/xinxinyang/data4/X.npy", X)
    np.save('/Users/xinxinyang/data4/Y.npy',Y)

#save_file()

def softmax(pred):
    for p in pred:
        if p<= 0.3:
            p=0
        elif p>=0.7 and p<= 1.3:
            p=1
        else:
            p=p
    return pred

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll

def classify():
    X=np.load("/Users/xinxinyang/data4/X.npy")
    Y=np.load("/Users/xinxinyang/data4/Y.npy")
    #Y = np.array(Y)
    nsamples, nx, ny,nz = X.shape
    d2_X = X.reshape((nsamples, nx * ny * nz))
    X_train, X_test, y_train, y_test = \
        train_test_split(d2_X, Y,test_size=0.2, random_state=20, stratify=Y)

    # Multilayer Perceptron
    print('Training Multilayer Perceptron ...')
    mlp_model = MLPClassifier(hidden_layer_sizes=(600, 800, 300),
            max_iter=600,verbose=True)
    mlp_model.summary()

    mlp_model.fit(X_train, y_train)

    mlp_y_train_pred = mlp_model.predict(X_train)
    mlp_y_test_pred = mlp_model.predict(X_test)
    print('Multilayer Perceptron train accuracy: ' + str(round(accuracy_score(
        y_train,mlp_y_train_pred),4)))
    print('Multilayer Perceptron test accuracy: ' + str(round(accuracy_score(
        y_test,mlp_y_test_pred),4)))
    print('Multilayer Perceptron train report:')
    print(classification_report(y_train, mlp_y_train_pred))
    print('Multilayer Perceptron test report:')
    print(classification_report(y_test, mlp_y_test_pred))

    mlp_probs = mlp_model.predict_proba(X_test)
    mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_probs[:, 1])
    mlp_roc_auc = auc(mlp_fpr, mlp_tpr)

    plt.plot(mlp_fpr, mlp_tpr, lw=2, label='Multilayer Perceptron (area = '
                                                   '%0.3f)' % mlp_roc_auc,
             color='orange')

    # random forest
    print('Training random forest')
    RF_model = RF(n_estimators=50, n_jobs=3)
    RF_model.fit(X_train, y_train)

    RF_y_train_pred = RF_model.predict(X_train)
    RF_y_test_pred = RF_model.predict(X_test)
    print('random forest train accuracy: ' + str(round(accuracy_score(y_train,
                                                  RF_y_train_pred),4)))
    print('random forest test accuracy: ' + str(round(accuracy_score(y_test,
                                                               RF_y_test_pred),4)))
    print('random forest train report:')
    print(classification_report(y_train,RF_y_train_pred))
    print('random forest test report:')
    print(classification_report(y_test,RF_y_test_pred))

    RF_probs = RF_model.predict_proba(X_test)
    RF_fpr, RF_tpr, _ = roc_curve(y_test, RF_probs[:,1])
    RF_roc_auc = auc(RF_fpr, RF_tpr)

    plt.plot(RF_fpr, RF_tpr, lw=2, label='random forest (area = '
                                                   '%0.3f)' % RF_roc_auc,
             color='blue')

    # XGBoost
    print('Training XGBoost ...')
    XGBoost_model = xgb.XGBClassifier(max_depth =
                                      10, objective="binary:logistic")
    XGBoost_model.fit(X_train, y_train)

    XGBoost_y_train_pred = XGBoost_model.predict(X_train)
    XGBoost_y_test_pred = XGBoost_model.predict(X_test)
    print('XGBoost train accuracy: ' + str(round(accuracy_score(y_train,
                                                  XGBoost_y_train_pred),4)))
    print('XGBoost test accuracy: ' + str(round(accuracy_score(y_test,
    XGBoost_y_test_pred),4)))

    print('XGBoost train report:')
    print(classification_report(y_train,XGBoost_y_train_pred))
    print('XGBoost test report:')
    print(classification_report(y_test, XGBoost_y_test_pred))

    XGBoost_probs = XGBoost_model.predict_proba(X_test)
    XGBoost_probs = XGBoost_probs[:, 1]

    XGBoost_fpr, XGBoost_tpr, _ = roc_curve(y_test, XGBoost_probs)
    roc_auc = auc(XGBoost_fpr, XGBoost_tpr)

    plt.plot(XGBoost_fpr, XGBoost_tpr, lw =2, label='XGBoost (area = '
                                                '%0.3f)' % roc_auc, color = 'red')

    # # SVM
    print("Training SVM with rbf kernel ... ")
    svm_model = svm.SVC(kernel='rbf', probability=True)
    print("1")
    svm_model.fit(X_train,y_train)
    print("2")

    svm_y_train_pred = svm_model.predict(X_train)
    print("3")
    svm_y_test_pred = svm_model.predict(X_test)
    print("4")
    print('SVM train accuracy: ' + str(round(accuracy_score(y_train,
                                                      svm_y_train_pred),4)))
    print('SVM test accuracy: ' + str(round(accuracy_score(y_test,
                                                     svm_y_test_pred),4)))
    print('SVM train report:')
    print(classification_report(y_train,  svm_y_train_pred))
    print('SVM test report:')
    print(classification_report(y_test,  svm_y_test_pred))

    svm_probs = svm_model.predict_proba(X_test)
    svm_probs = svm_probs[:, 1]

    svm_fpr,  svm_tpr, _ = roc_curve(y_test,  svm_probs)
    svm_roc_auc = auc( svm_fpr,  svm_tpr)

    plt.plot( svm_fpr,  svm_tpr, lw=2, label='SVM-linear kernal (area = '
                                                   '%0.3f)' %  svm_roc_auc,
             color='green')

    # PCA + SVM
    print('Training PCA + SVM ...')
    pca = PCA(n_components=300, whiten=True, random_state=42)
    svc = svm.SVC(kernel='rbf',probability=True)

    pipe = make_pipeline(pca, svc)
    pipe.fit(X_train, y_train)

    pipe_y_train_pred = pipe.predict(X_train)
    pipe_y_test_pred = pipe.predict(X_test)
    print('PCA + SVM train accuracy: ' + str(round(accuracy_score(y_train,
                                                      pipe_y_train_pred),4)))
    print('PCA + SVM test accuracy: ' + str(round(accuracy_score(y_test,
                                                     pipe_y_test_pred),4)))
    print('PCA + SVM train report:')
    print(classification_report(y_train, pipe_y_train_pred))
    print('PCA + SVM test report:')
    print(classification_report(y_test, pipe_y_test_pred))

    pipe_probs = pipe.predict_proba(X_test)
    pipe_probs = pipe_probs[:, 1]

    pipe_fpr, pipe_tpr, _ = roc_curve(y_test, pipe_probs)
    pipe_roc_auc = auc(pipe_fpr, pipe_tpr)

    plt.plot(pipe_fpr, pipe_tpr, lw=2, label='PCA + SVM (area = '
                                           '%0.3f)' % pipe_roc_auc,
             color='yellow')


    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.show()

classify()
