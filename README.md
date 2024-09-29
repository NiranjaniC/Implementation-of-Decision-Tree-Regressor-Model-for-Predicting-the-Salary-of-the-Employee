# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 

2.Upload the dataset and check for any null values using .isnull() function. 

3.Import LabelEncoder and encode the dataset. 

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset. 

5.Predict the values of arrays. 

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset. 

7.Predict the values of array. 8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Niranjani.C
RegisterNumber:  212223220069

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

dt.predict([[5,6]])plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

*/
```

## Output:
![Screenshot 2024-09-29 181117](https://github.com/user-attachments/assets/abe20e56-fbf7-4f13-b0c8-5eaef9ee55c0)
![Screenshot 2024-09-29 181611](https://github.com/user-attachments/assets/2f3fcfc7-db73-46ab-bde4-02851bf5abfe)
![Screenshot 2024-09-29 181708](https://github.com/user-attachments/assets/c33456bd-2fa3-48f8-a4dd-34de7bad3800)
![Screenshot 2024-09-29 181718](https://github.com/user-attachments/assets/269ff8ec-f595-4cb1-857e-345d9eb1afd9)
![Screenshot 2024-09-29 181727](https://github.com/user-attachments/assets/4bc12057-b96c-437a-ac8c-6949ff9eb089)
![Screenshot 2024-09-29 181734](https://github.com/user-attachments/assets/eb89f42e-159a-4c8e-8625-6b731985d4c9)
![Screenshot 2024-09-29 181815](https://github.com/user-attachments/assets/36d37b38-58ff-4f0e-99cb-f89eb3024c41)
![Screenshot 2024-09-29 181831](https://github.com/user-attachments/assets/47234175-3b80-4333-b0ec-cb84b91638e5)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
