import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.svm import SVC
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
warnings.warn = warn #disable the unnecessary warning caused by ill-defined calculation e.g. 1/0, in the function classification_report
A_x0 = []
#name_list = ["breast-cancer","diabetes","digit","iris","wine"]
name_list = ["breast-cancer","diabetes","digit","iris","wine"]

z = zipfile.ZipFile("datasets.zip", "r")
if os.path.isdir("datasets"):  
    pass  
else:  
    os.mkdir("datasets")  
for name in name_list:  
    z.extract("datasets/"+name+".npz","./")  
z.close()

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
    model = SVC()
#===========================================================================================================
#                  tune the value of gamma and use the best paramemter to train and test automatically
#===========================================================================================================
    print("tuning parameters for the dataset "+"\033[1;31m%s\033[0m" % name+" ..., which may take tens of seconds")
    parameters = {'gamma':[1,0.1,0.01,0.001]} # tune the best number of neurons in the hidden layer
    clf = GridSearchCV(model, parameters,scoring='accuracy',cv=5)
    clf.fit(X, Y) #training with different paramemters
    print("finish tuning")
#===========================================================================================================
#      train once more with the same parameters to collect total time of learning withou predicting
#===========================================================================================================
    time_start=time.time()   
    new_model = SVC(gamma=clf.best_params_['gamma'])
    new_model.fit(X, Y)
    time_end=time.time()   
    time_cost = (time_end-time_start)*1000
#==================================================================================================
#                  print some statistics on terminal
#==================================================================================================
    A_x0.append(sklearn.metrics.accuracy_score(Y_test, clf.predict(X_test)))
    print("The"+"\033[1;31m%s\033[0m" %" Candidate parameters"+" and corresponding accuracy for "+"\033[1;31m%s\033[0m" % name +" are shown below.")
    print("Gamma    =  "+'1'.center(7)+'|'+'0.1'.center(7)+'|'+'0.01'.center(7)+'|'+'0.001'.center(7)+'|') # accuracy for different parameters
    print("Accuracy = ", end="  ")
    for item in clf.cv_results_['mean_test_score']:
        print("%.4f"%item, end="| ")
    print()
    print("The"+"\033[1;31m%s\033[0m" %" best parameters"+" selected by scikit-learn for "+"\033[1;31m%s\033[0m" % name +" are shown below.")
    print(clf.best_params_) # the best parameters
    print("Time required by one procedure of RBF SVM is "+"\033[1;31m%s\033[0m" % '%.2f'% time_cost + " ms.")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the RBF SVC model on the "+"\033[1;31m%s\033[0m"%"training sets"+" is "+str(sklearn.metrics.accuracy_score(Y, new_model.predict(X))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the RBF SVC model on the "+"\033[1;31m%s\033[0m"%"training sets"+"     is "+str(sklearn.metrics.log_loss(Y, new_model.predict(X)))+".")
    print("The"+"\033[1;31m%s\033[0m"%" accuracy"+" of the RBF SVC model on the "+"\033[1;31m%s\033[0m"%"test sets"+"     is "+str(sklearn.metrics.accuracy_score(Y_test, new_model.predict(X_test))))
    print("The"+"\033[1;31m%s\033[0m"%" loss"+" of the RBF SVC model on the "+"\033[1;31m%s\033[0m"%"test sets"+"         is "+str(sklearn.metrics.log_loss(Y_test, new_model.predict(X_test)))+".")
    print("The"+"\033[1;31m%s\033[0m" %" classification report of test"+" for ["+"\033[1;31m%s\033[0m" % name +"] are shown below.")
    print(sklearn.metrics.classification_report(Y_test, new_model.predict(X_test)))
    print("=========================================================================")

#===========================================================================================================
#                  plot some statistics as bar chart
#===========================================================================================================
n = 5
fig = plt.figure(1,figsize=(8,4))  
ax  = fig.add_subplot(111) 
bar_X = np.arange(n)+1
ax.set_title('Accuracy for Datasets based on RBF SVC')
ax.set_xlabel('dataset')
ax.set_ylabel('accuracy')
ax.set_xticks(bar_X)  
ax.set_xticklabels(name_list)  

print("======================== Bar Chart of Precision =========================\n")
plt.bar(bar_X, A_x0, alpha=0.9, width = 0.7, color = ['lightskyblue','blanchedalmond','firebrick','yellow','yellowgreen'], edgecolor = 'black', label='Accuracy', lw=1)
for x,y in zip(bar_X,A_x0): plt.text(x, y+0.0, '%.2f' % y, ha='center', va= 'bottom') 
plt.ylim(0.0,+1.05)
plt.grid(True)  
plt.show()

