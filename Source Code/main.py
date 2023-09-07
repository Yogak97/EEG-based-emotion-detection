##Emotion detections using EEG signals and Emotive insight low cost wireless headset device [BCI]
# Author : Yoga K
# Last Modified : June 03 2019 6:40:07
##################Analysis and Results On Trained Models####################



import pandas as pd  
import numpy as np   
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split  
from sklearn import svm  
from sklearn.metrics import classification_report, confusion_matrix  
 
print("Hii !!!!!!Welcome to our System\n")
 
train = pd.read_csv("HappySadC.csv")
X= train.drop('Label', axis=1)  
Y = train['Label']  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20) 


#svclassifier = SVC(kernel='linear')  
#svclassifier.fit(X_train,y_train)  

####################SVM#############################
print("--------------------SVM Results--------------------------")
classifier = svm.SVC(kernel='rbf',gamma=0.01)
model=classifier.fit(X_train, y_train) 

#test = pd.read_csv("v4h.csv")
#x_test= test.drop('Label', axis=1) 
#y_test = test['Label']  


y_pred = classifier.predict(X_test)
print(y_pred)

print("\n\nAccuracy of model is:")
print(model.score(X_test,y_test)*100,"%")

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))
 

#cross validation
scores=cross_val_score(classifier,X,Y,cv=6)
print("cross validated scores are:\n",scores)
print()
print("confusion matrix is:")
print(confusion_matrix(y_test,y_pred)) 
print("classification report is:")
print(classification_report(y_test,y_pred)) 



####################DT#############################
print("--------------------Decision Tree Results--------------------------")

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()

model=classifier.fit(X_train, y_train) 

print(model.score(X_test,y_test)*100,"%")
 

y_pred = classifier.predict(X_test)

#cross validation
scores=cross_val_score(classifier,X,Y,cv=6)
print("cross validated scores are:\n",scores)
print()
print("confusion matrix is:")
print(confusion_matrix(y_test,y_pred)) 
print("classification report is:")
print(classification_report(y_test,y_pred)) 



####################KNN#############################
print("--------------------------KNN-------------------------------------")
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=2)
model=classifier.fit(X,Y)

y_pred=model.predict(X_test)
print(y_pred)
print(model.score(X_test,y_test)*100,"%%%%%%")

#cross validation
scores=cross_val_score(classifier,X,Y,cv=6)
print("cross validated scores are:\n",scores)
print()
print("confusion matrix is:")
print(confusion_matrix(y_test,y_pred)) 
print("classification report is:")
print(classification_report(y_test,y_pred)) 



####################NB#############################
print("----------------------------NB------------------------------------------")

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

model=classifier.fit(X,Y)

y_pred=model.predict(X_test)
print(y_pred)
print(model.score(X_test,y_test)*100,"%%%%%%")
#cross validation
scores=cross_val_score(classifier,X,Y,cv=6)
print("cross validated scores are:\n",scores)
print()
print("confusion matrix is:")
print(confusion_matrix(y_test,y_pred)) 
print("classification report is:")
print(classification_report(y_test,y_pred)) 



####################Feature Importance##############################

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(25).plot(kind='barh')


#######################Prediction Among Two States##################

print("\nHi! Welcome to our System for Predicting Emotions :\n")

test = pd.read_csv("geetaH.csv")  #Unseen data not used in training h=happy , s=sad
X_test= test.drop('Label', axis=1) 
y_test= test['Label']

print("\n\nWe are working to predict your mood")

#######################SVM Prediction#########################
classifier = svm.SVC(kernel='rbf')
print("fitting") 
model=classifier.fit(X, Y) 
y_pred = classifier.predict(X_test)
print(y_pred)

h=0
s=0
for i in y_pred:
	if(i=="sad"):
		s+=1
	if(i=="happy"):
		h+=1
			
if(h>s):
	print("\n\nyehhh....It seems that you are happy man")
else:
	print("\n\noooops...Bro you are lokking little sad\n\n")	


print("bye")

#######################KNN Prediction#########################
print("------------------------KNN-----------------------")
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=2)
model=classifier.fit(X,Y)
y_pred=model.predict(X_test)
print(y_pred)

h=0
s=0
for i in y_pred:
	if(i=="sad"):
		s+=1
	if(i=="happy"):
		h+=1
			
if(h>s):
	print("\n\nyehhh....It seems that you are happy man")
else:
	print("\n\noooops...Bro you are lokking little sad\n\n")	

#######################DT Prediction##################################

print("----------------------------------Decision tree---------------------")
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
model=classifier.fit(X,Y) 
y_pred = classifier.predict(X_test)
print(y_pred)


h=0
s=0
for i in y_pred:
	if(i=="sad"):
		s+=1
	if(i=="happy"):
		h+=1
			
if(h>s):
	print("\n\nyehhh....It seems that you are happy man")
else:
	print("\n\noooops...Bro you are lokking little sad\n\n")	

################################################################
plt.show()


 





