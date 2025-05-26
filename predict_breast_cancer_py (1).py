!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#upload downloaded kaggle library in colab
df=pd.read_csv('data.csv')

#first 5 rows
df.head()

#EDA
#checking the total no. of rows and columns
df.shape

#checking the columns and corresponding data types
#the properties of data - summary statistics
df.info()

#2nd way to check for null values
df.isnull().sum()

#drop the coulums with all missing values
df=df.dropna(axis=1)

df.shape

#checking for datatypes
df.dtypes

#data visualization
df['diagnosis'].value_counts()

sns.countplot(x=df['diagnosis'], palette=['orange', 'blue'], label='count')

#tranform categorical to numerical
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df['diagnosis'] = labelencoder_Y.fit_transform(df['diagnosis'])  # 0 = Benign, 1 = Malignant

df.iloc[:,1].values

sns.pairplot(df.iloc[:,1:7], hue= 'diagnosis')

#correlation between columns
df.iloc[:,1:11].corr()

#heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:11].corr(), cmap= "YlGnBu", annot= True, fmt= ".0%")

#feature scaling
#split our data set into independent and dependent datasets
#independent--> X
#dependent--> Y
X = df.iloc[:, 2:31].values
Y = df['diagnosis'].values

#80:20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size= 0.20, random_state= 0)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # Don’t use fit_transform on test

Y_train = Y_train.ravel()  # Flatten if needed

def models(X_train, Y_train):
  from sklearn.linear_model import LogisticRegression
  log= LogisticRegression(random_state= 0)
  log.fit(X_train, Y_train)

  from sklearn.tree import DecisionTreeClassifier
  tree= DecisionTreeClassifier(criterion= 'entropy', random_state= 0)
  tree.fit(X_train, Y_train)

  from sklearn.ensemble import RandomForestClassifier
  forest= RandomForestClassifier(n_estimators= 10, criterion= 'entropy', random_state= 0)
  forest.fit(X_train, Y_train)

  #print the accuracy of each model on the training dataset
  print('The accuracy of Logistic Regression: ',log.score(X_train, Y_train))
  print('The accuracy of Decision Tree: ',tree.score(X_train, Y_train))
  print('The accuracy of Random Forest: ',forest.score(X_train, Y_train))

  return log, tree, forest

model = models(X_train, Y_train)

"""Now that the models are trained, you can evaluate their performance on the test data and make predictions."""

#evaluate the performance of model
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, model[0].predict(X_test))
tp= cm[0][0]
tn= cm[1][0]
fn= cm[1][1]
fp= cm[0][1]
print(cm)
print('Acurracy: ',(tp+tn)/(tp+tn+fn+fp))

#Model accuracy on confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('Model ', i)
  print(classification_report(Y_test, model[i].predict(X_test)))
  print(accuracy_score(Y_test, model[i].predict(X_test)))
  print()

#Model Prediction vs Actual Prediction

#prediction
pred= model[2].predict(X_test)
print('Our Model Prediction')
print(pred)
print()
#actual
print('Actual Prediction')
print(Y_test)
