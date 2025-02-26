# EXPERIMENT 02: IMPLEMENTATION OF SIMPLE LINEAR REGRESSION MODEL FOR PREDICTING THE MARKS SCORED
## AIM:  
To write a program to predict the marks scored by a student using the simple linear regression model.
## EQUIPMENTS REQUIRED:
1. Hardware – PCs  
2. Anaconda – Python 3.7 Installation / Jupyter notebook  
## ALGORITHM:
1.Import the required libraries and read the dataframe.  
2.Assign hours to X and scores to Y.  
3.Implement training set and test set of the dataframe.  
4.Plot the required graph both for test data and training data.  
5.Find the values of MSE , MAE and RMSE.     

## PROGRAM:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GOKHULRAJ V
RegisterNumber:  212223230064
```
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

#Displaying the content in datafile
df.head()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/e5f39f33-fe67-49a4-ae7d-8d8498d3c5e7)
```py
#Last five rows
df.tail()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/022eb321-7225-4178-9945-6cbaeb3d334f)

```py
#Segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/43a5570d-6916-4e57-b232-7f445a2f7413)

![image](https://github.com/user-attachments/assets/0f822094-b2f8-400b-9599-74e770df411f)

```py
#Splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#Displaying predicted values
Y_pred
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/7d746620-a53a-4271-9077-ffb5f2073f40)
```py
#Displaying actual values
Y_test
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/8a445cc3-07fb-4837-afca-f4ac9636208c)
```py
#Graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/230e2f4e-8c80-4732-974e-d22b81652a2a)
```py
#Graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="orange")
plt.title("Hours VS Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/7c15a489-216e-46db-85f9-84a39c2266f2)
```py
#MSE
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

#MAE
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

#RMSE
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
````
## OUTPUT:
![image](https://github.com/user-attachments/assets/d39891b9-be73-4b01-9ae0-d61bdc2573a9)


## RESULT:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

