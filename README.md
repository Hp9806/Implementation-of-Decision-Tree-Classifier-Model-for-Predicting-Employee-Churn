# EX:6 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Developed by: S Harish Kumar.
RegisterNumber:212221230104.
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/8d7471da-0948-48c2-ae45-15fec635f32e)


![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/8b7b8ced-e77b-46ce-809c-981e3b7c63c0)



![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/e26b5e3d-a44a-4b77-a93e-0607996726ea)


![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/1aa4048a-3ef5-4d18-82fb-ce444e9123a5)


![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/6707435e-2170-4f6e-9430-fba8b4c0a628)


![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/18b510a2-f585-4a0d-a1e3-89dc0f3d1740)



![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/5b06aada-ff32-4aef-97ff-8258596d29b7)



![image](https://github.com/Adithya-Siddam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427248/667ce484-01d1-4748-bf97-8e455156aa53)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
