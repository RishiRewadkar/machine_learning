import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

data = pd.read_csv('Dataset.csv',sep=",")
data=data.drop(columns='ID')
le=LabelEncoder()
data['Age']=le.fit_transform(data['Age'])
data['Income']=le.fit_transform(data['Income'])
data['Gender']=le.fit_transform(data['Gender'])
data['Marital Status']=le.fit_transform(data['Marital Status'])
data['Buys']=le.fit_transform(data['Buys'])

x=data.drop(columns='Buys')
y=data['Buys']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

classifier=DecisionTreeClassifier(criterion='entropy',max_depth=4)
classifier.fit(x_train,y_train)

accuracy=classifier.score(x_test,y_test)
print(accuracy)
