import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.svm import LinearSVC
import time
import zipfile
import os

def warn(*args, **kwargs):
    pass
import warnings

name_list = ["breast-cancer","diabetes","digit","iris","wine"]

z = zipfile.ZipFile("datasets.zip", "r")
if os.path.isdir("datasets"):  
    pass  
else:  
    os.mkdir("datasets")  
for name in name_list:  
    z.extract("datasets/"+name+".npz","./")  
z.close() 

A_x0 = []

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
#                  initial model with default parameters
#===========================================================================================================
    model = LinearSVC()
#===========================================================================================================
#                  traing the model with default parameters
#===========================================================================================================
    model.fit(X, Y)
    time_end=time.time()
#===========================================================================================================
#                  print some statistics on terminal
#===========================================================================================================

    time_cost = (time_end-time_start)*1000
    A_x0.append(sklearn.metrics.accuracy_score(Y_test, model.predict(X_test)))
    print("The"+"\033[1;31m%s\033[0m" %" result"+" collected by scikit-learn for "+"\033[1;31m%s\033[0m" % name +" are shown below.")
    print("Time required by logistic regression is "+"\033[1;31m%s\033[0m" % '%.2f'% time_cost + " ms.")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the Linear SVC model on the "+"\033[1;31m%s\033[0m"%"training sets"+" is "+str(sklearn.metrics.accuracy_score(Y, model.predict(X))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the Linear SVC model on the "+"\033[1;31m%s\033[0m"%"training sets"+"     is "+str(sklearn.metrics.log_loss(Y, model.predict(X)))+".")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the Linear SVC model on the "+"\033[1;31m%s\033[0m"%"test sets"+"     is "+str(sklearn.metrics.accuracy_score(Y_test, model.predict(X_test))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the Linear SVC model on the "+"\033[1;31m%s\033[0m"%"test sets"+"         is "+str(sklearn.metrics.log_loss(Y_test, model.predict(X_test)))+".")
    print("The"+"\033[1;31m%s\033[0m" %" classification report of test"+" for ["+"\033[1;31m%s\033[0m" % name +"] are shown below.")
    print(sklearn.metrics.classification_report(Y_test, model.predict(X_test)))
    print("=========================================================================")
#===========================================================================================================
#                  plot some statistics as training curve
#===========================================================================================================
n = 5
fig = plt.figure(1,figsize=(8,4))  
ax  = fig.add_subplot(111) 
bar_X = np.arange(n)+1
ax.set_title('Accuracy for Datasets based on LinearSVC')
ax.set_xlabel('dataset')
ax.set_ylabel('accuracy')
ax.set_xticks(bar_X)  
ax.set_xticklabels(name_list)  

print("======================== Bar Chart of Accuracy =========================\n")
plt.bar(bar_X, A_x0, alpha=0.9, width = 0.7, color = ['lightskyblue','blanchedalmond','firebrick','yellow','yellowgreen'], edgecolor = 'black', label='Accuracy', lw=1)
for x,y in zip(bar_X,A_x0): plt.text(x, y+0.0, '%.2f' % y, ha='center', va= 'bottom') 
plt.ylim(0.0,+1.05)
plt.grid(True)  
plt.show()

