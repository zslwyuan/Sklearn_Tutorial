import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sklearn
import time
import math
import zipfile
import os

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn #ignore the FutureWarning caused by default paramemters in scikit-learn

name_list = ["breast-cancer","diabetes","digit","iris","wine"]

z = zipfile.ZipFile("datasets.zip", "r")
if os.path.isdir("datasets"):  
    pass  
else:  
    os.mkdir("datasets")  
for name in name_list:  
    z.extract("datasets/"+name+".npz","./")  
z.close()  

fig_array = [321,322,323,324,325]
fig_num = 0
plt.figure(1,figsize=(10,10))

print("====================== Statics of Different Datasets ==============================\n")
for name in name_list:
#===========================================================================================================
#                  load data
#===========================================================================================================
    data = np.load("datasets/"+name+".npz")
    X = data['train_X']       #load data for training
    Y = data['train_Y']
    X_test = data['test_X']   #load data for testing
    Y_test = data['test_Y']
#===========================================================================================================
#                  scale data
#===========================================================================================================
    scaler = StandardScaler()
    scaler.fit(X)  # fit on training data
    X = scaler.transform(X) #scale the data
    X_test = scaler.transform(X_test)  # apply same transformation to test data
#===========================================================================================================
#                  initial model
#===========================================================================================================
    model = MLPClassifier()
    parameters = {'activation':['logistic'], 'hidden_layer_sizes':[1,2,3,4,5,6,7,8,9,10], \
                  'max_iter':[10000]} # tune the best number of neurons in the hidden layer
#===========================================================================================================
#                  tune the number of hidden units for the layer
#===========================================================================================================
    print("tuning parameters for the dataset "+"\033[1;31m%s\033[0m" % name+" ..., which may take tens of seconds")
    clf = GridSearchCV(model, parameters,scoring='accuracy',cv=5)
    clf.fit(X, Y) #training with different paramemters
    print("finish tuning, begin to use best paramemters to train and test")
#===========================================================================================================
#                  use the best parameter to train and test (to plot the training curve)
#===========================================================================================================
    time_start=time.time() 
    best_model =  MLPClassifier(activation='logistic', hidden_layer_sizes=clf.best_params_['hidden_layer_sizes'],max_iter=10000)

    A_training = [] 
    A_test = []
    TIME_STAMP = []
    mini_batch_size = int(len(X)/20)
    epoch = 150  
    
    best_model.partial_fit(X[0:1], Y[0:1], classes=np.unique(Y))
    for ep in range(epoch):
        for i in range(0,len(X),mini_batch_size):
            if ((i+mini_batch_size)<len(X)):
                best_model.partial_fit(X[i:(i+mini_batch_size)], Y[i:(i+mini_batch_size)])
            else:
                best_model.partial_fit(X[i:len(X)], Y[i:len(X)])
        A_training.append (sklearn.metrics.accuracy_score(Y, best_model.predict(X)))
        A_test.append(sklearn.metrics.accuracy_score(Y_test, best_model.predict(X_test)))
        TIME_STAMP.append(time.time()-time_start)
#===========================================================================================================
#      use the best parameter to train once more (to collect the total training time without predicting)
#===========================================================================================================
    time_start=time.time() 
    best_model =  MLPClassifier(activation='logistic', hidden_layer_sizes=clf.best_params_['hidden_layer_sizes'],max_iter=10000)
    
    best_model.partial_fit(X[0:1], Y[0:1], classes=np.unique(Y))
    for ep in range(epoch):
        for i in range(0,len(X),mini_batch_size):
            if ((i+mini_batch_size)<len(X)):
                best_model.partial_fit(X[i:(i+mini_batch_size)], Y[i:(i+mini_batch_size)])
            else:
                best_model.partial_fit(X[i:len(X)], Y[i:len(X)])
    time_cost =time.time() - time_start
#===========================================================================================================
#                  plot some statistics as training curve
#===========================================================================================================
    colors = ['r','y']
    legends = {'Training':'o','Test':'*'}
    plt.subplot(fig_array[fig_num])
    plt.plot(TIME_STAMP,A_training,'r-')
    plt.plot(TIME_STAMP,A_test,'b-.')
    plt.legend(legends.keys(),loc=4)
    plt.grid()
    plt.xlabel('time(s)')
    plt.ylabel('accuracy')
    plt.title("Performance over time for "+name)    
    fig_num = fig_num+1
#===========================================================================================================
#                  print some statistics on terminal
#===========================================================================================================
    print("The"+"\033[1;31m%s\033[0m" %" Candidate parameters"+" and corresponding accuracy for "+"\033[1;31m%s\033[0m" % name +" are shown below.")
    print("H        =  "+'1'.center(7)+'|'+'2'.center(7)+'|'+'3'.center(7)+'|'+'4'.center(7)+'|'+'5'.center(7)+'|'+'6'.center(7)+'|'+'7'.center(7)+'|'+'8'.center(7)+'|'+'9'.center(7)+'|'+'10'.center(7)+'|') # accuracy for different parameters
    print("Accuracy = ", end="  ")
    for item in clf.cv_results_['mean_test_score']:
        print("%.4f"%item, end="| ")
    print()
    print("The"+"\033[1;31m%s\033[0m" %" best parameters"+" selected by scikit-learn for "+"\033[1;31m%s\033[0m" % name +" are shown below.")
    print(clf.best_params_) # the best parameters
    print("The"+"\033[1;31m%s\033[0m" %" training time"+" collected by scikit-learn       is "+"\033[1;31m%s\033[0m" %"%.2f"%time_cost +" s.")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the MLP model on the "+"\033[1;31m%s\033[0m"%"training sets"+" is "+str(sklearn.metrics.accuracy_score(Y, clf.predict(X))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the MLP model on the "+"\033[1;31m%s\033[0m"%"training sets"+"     is "+str(sklearn.metrics.log_loss(Y, clf.predict(X)))+".")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the MLP model on the "+"\033[1;31m%s\033[0m"%"test sets"+"     is "+str(sklearn.metrics.accuracy_score(Y_test, clf.predict(X_test))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the MLP model on the "+"\033[1;31m%s\033[0m"%"test sets"+"         is "+str(sklearn.metrics.log_loss(Y_test, clf.predict(X_test)))+".")
    print("The"+"\033[1;31m%s\033[0m" %" classification report of test"+" for ["+"\033[1;31m%s\033[0m" % name +"] are shown below.")
    print(sklearn.metrics.classification_report(Y_test, clf.predict(X_test)))
    print("=========================================================================")
  
plt.tight_layout()
plt.show()

