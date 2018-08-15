import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
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
    time_start=time.time() 
#===========================================================================================================
#                  initial model and related statistic arrays
#===========================================================================================================
    model = linear_model.SGDClassifier(learning_rate='invscaling',eta0=0.01,power_t=0.5,loss='log')   #logistic regression model
    A_training = [] 
    A_test = []
    TIME_STAMP = []
    mini_batch_size = min(200,len(X))
    epoch = 300
#===========================================================================================================
#                  learning based on partial_fit and mini_batch
#===========================================================================================================
    model.partial_fit(X[0:1], Y[0:1], classes=np.unique(Y))
    for ep in range(epoch):
        for i in range(0,len(X),mini_batch_size):
            if ((i+mini_batch_size)<len(X)):
                model.partial_fit(X[i:(i+mini_batch_size)], Y[i:(i+mini_batch_size)])
            else:
                model.partial_fit(X[i:len(X)], Y[i:len(X)])
            model.partial_fit(X[i:(i+mini_batch_size)], Y[i:(i+mini_batch_size)])
        A_training.append (sklearn.metrics.accuracy_score(Y, model.predict(X)))
        A_test.append(sklearn.metrics.accuracy_score(Y_test, model.predict(X_test)))
        TIME_STAMP.append(time.time()-time_start)
#===========================================================================================================
#      train once more with the same parameters to collect total time of learning withou predicting
#===========================================================================================================
    time_start=time.time()
    model = linear_model.SGDClassifier(learning_rate='invscaling',eta0=0.01,power_t=0.5,loss='log')   #logistic regression model
    model.partial_fit(X[0:1], Y[0:1], classes=np.unique(Y))
    for ep in range(epoch):
        for i in range(0,len(X),mini_batch_size):
            if ((i+mini_batch_size)<len(X)):
                model.partial_fit(X[i:(i+mini_batch_size)], Y[i:(i+mini_batch_size)])
            else:
                model.partial_fit(X[i:len(X)], Y[i:len(X)])
            model.partial_fit(X[i:(i+mini_batch_size)], Y[i:(i+mini_batch_size)])
    time_cost = (time.time()-time_start)*1000
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
#==================================================================================================
#                  print some statistics on terminal
#==================================================================================================
    print("The"+"\033[1;31m%s\033[0m" %" result"+" collected by scikit-learn for ["+"\033[1;31m%s\033[0m" % name +"] are shown below.")
    print("The"+"\033[1;31m%s\033[0m" %" training time"+" collected by scikit-learn       is "+"\033[1;31m%s\033[0m" %"%.2f"%time_cost +" ms.")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the LR model on the "+"\033[1;31m%s\033[0m"%"training sets"+" is "+str(sklearn.metrics.accuracy_score(Y, model.predict(X))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the LR model on the "+"\033[1;31m%s\033[0m"%"training sets"+"     is "+str(sklearn.metrics.log_loss(Y, model.predict(X)))+".")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the LR model on the "+"\033[1;31m%s\033[0m"%"test sets"+"     is "+str(sklearn.metrics.accuracy_score(Y_test, model.predict(X_test))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the LR model on the "+"\033[1;31m%s\033[0m"%"test sets"+"         is "+str(sklearn.metrics.log_loss(Y_test, model.predict(X_test)))+".")
    print("The"+"\033[1;31m%s\033[0m" %" classification report of test"+" for ["+"\033[1;31m%s\033[0m" % name +"] are shown below.")
    print(sklearn.metrics.classification_report(Y_test, model.predict(X_test)))
    print("=========================================================================")

plt.tight_layout()
plt.show()

