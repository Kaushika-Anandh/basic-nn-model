Exp.No : 01 
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
Date : 16.08.2023
<br>
# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

- Neural networks consist of simple input/output units called neurons. 
- Regression helps in establishing a relationship between a dependent variable and one or more independent variables.
- Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.
- Build your training and test set from the dataset, here we are making the neural network 2 hidden layers with activation function as relu and with their nodes in them.
- Then we will fit our dataset and then predict the value.

## Neural Network Model

<p align="center">
<img src="https://github.com/Kaushika-Anandh/basic-nn-model/blob/main/network%20model.PNG" width="650" height="400">
</p>

<br>

## DESIGN STEPS

- **Step 1:** Load the dataset fron authorizing the access to drive or directly ope using read.csv
- **Step 2:** Split the dataset into training and testing
- **Step 3:** Create MinMaxScalar objects ,fit the model and transform the data.
- **Step 4:** Build the Neural Network Model and compile the model.
- **Step 5:** Train the model with the training data.
- **Step 6:** Plot the performance plot
- **Step 7:** Evaluate the model with the testing data.

## PROGRAM

> Developed by: Kaushika A <br>
> Register no: 212221230048

**reading the data file**
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)

sheet=gc.open('exp 1 dataset').sheet1
data=sheet.get_all_values()
```

**importing packages**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
```

**creating dataframe**
```python
df=pd.DataFrame(data[1:],columns=data[0])
df=df.astype({'input':'float'})
df=df.astype({'output':'float'})
df.head()
```

**creating train & test data**
```python
X=df[['input']].values
y=df[['output']].values
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=33)
```

**scaling the data**
```python
scaler=MinMaxScaler()
scaler.fit(x_train)
x_trained=scaler.transform(x_train)
```

**creating and compiling the network model**
```python
ai_brain = Sequential([
    Dense(units = 10, activation = 'relu', input_shape=[1]),
    Dense(units = 19,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')
```

**training the network model**
```python
ai_brain.fit(x_trained,y_train,epochs=2000)
```

**plotting loss graph**
```python
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

**find RMSE of the network model**
```python
X_test1 = scaler.transform(x_test)
ai_brain.evaluate(X_test1,y_test)
```

**predicting using the network model**
```python
X_n1=[[4]]
X_n1_1=scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```


## Dataset Information

<img src="https://github.com/Kaushika-Anandh/basic-nn-model/blob/main/dataset.PNG" width="100" height="400">

## OUTPUT

### Training Loss Vs Iteration Plot

<img src="https://github.com/Kaushika-Anandh/basic-nn-model/blob/main/plot.png" width="400" height="300">

<br>

### Test Data Root Mean Squared Error

<img src="https://github.com/Kaushika-Anandh/basic-nn-model/blob/main/1.PNG" width="400" height="40">

<br>

### New Sample Data Prediction

<img src="https://github.com/Kaushika-Anandh/basic-nn-model/blob/main/2.PNG" width="300" height="150">

## RESULT
Thus a basic neural network regression model for the given dataset is written and executed successfully.
