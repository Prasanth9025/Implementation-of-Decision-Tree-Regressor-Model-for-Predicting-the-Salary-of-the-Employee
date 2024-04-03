# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the
6. Required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Prasanth U
RegisterNumber: 212222220031
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
data["Position"].value_counts()
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y=data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(x_test)
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))

plot_tree(dt,feature_names=x.columns, filled=True)
plt.show()
```
## Output:

![image](https://github.com/Prasanth9025/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343686/b233ee22-36e0-4448-ac27-45b53245ceb6)

![image](https://github.com/Prasanth9025/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343686/a52b1639-0922-421a-9663-24416f4f6793)

![image](https://github.com/Prasanth9025/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343686/e10d0a2a-0826-4cff-8ffa-bbfd20c5d698)

![image](https://github.com/Prasanth9025/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343686/1f9140f9-310b-4853-9f8b-214f41274bf8)

![image](https://github.com/Prasanth9025/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343686/4e913819-6918-4b59-9078-e05bd66ba1e2)

![image](https://github.com/Prasanth9025/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118343686/e7d70784-c786-44a8-a0e2-c56f2a6e8251)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
