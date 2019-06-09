import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

df=pd.read_csv("cancer.data",sep=",")
df.replace('?',-9999,inplace=True)  # replacing missing data with -99999 as outlier
df.drop(['id'],1,inplace=True)

x=df.drop(columns="class")
y=df["class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

clf=sklearn.linear_model.LogisticRegression()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)
