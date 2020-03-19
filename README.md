 



## Designing A Multivariate Timeseries model


## Summary
In a time series problem, we aim to predict the future based on the data provided in the past. Actually, considering the recent past, we are able to predict the sales or market in the future. In the dataset provided by Au-dela, 3 features are presented as Val_1, Val_2 and Val_3. Despite the few number of available samples in the dataset, some statistical methods employed to predict the 4 steps ahead of the sequence of the data points. It is worth to say that since the performance of system is not matter in the scope of this project, conducting a research on the most common available method for time series prediction is the priority of this report. In addition, the solution aims to illustrate the style of coding on python and using object-oriented programming with respect to solving time series forecasting. With this in mind, the provided solution in both report and code files is not the most effective and recommended algorithm, while deep learning methods such as recurrent neural network based on LSTM architecture would be better mean.


## Introduction
The dataset consists from 3 features, which cause our problem to be a multivariate timeseries. Generally, different methods are available to be deployed in order to predict the next sequences of data. Some of these classical methods are very close to statistical analysing such as Vector Auto Regressive (VAR), Auto Regressive (AR), ARMA and so on. These models are available using Statsmodels library in python. Also, some Machine Learning models such as GRU, Recurrent Neural Networks (RNN) and Long Short-Term Memories (LSTM) can be utilized as well. 
In this project the classical methods have been implemented due to lack of time, however one of the best options for this type of problem is LSTM. This solution will be addressed and implemented in further works.
The classical methods, which have been applied to this problem are listed as follows:
•	Vector Autoregression (VAR)
•	Vector Autoregression Moving-Average (VARMA)
•	Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX). 
In the remain of the report, first, a brief description on each method would be provided, then we will explain the data preparation step followed by the models, which have been exerted as well as the forecast of models performance. 
Classical Time series Models
Vector Autoregression (VAR)
This method models the next step in each time series using an AR model. In fact, it is the generalization of AR to multiple parallel time series and is suitable for multivariate time series without trend and seasonal components.
 
Vector Autoregression Moving-Average (VARMA)
VARMA method models the next step in each time series using an ARMA model and is suitable for multivariate time series without trend and seasonal components.
 
Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)
This method is an extension of VARMA that includes modeling of exogenous variables as well. It is suitable for multivariate time series without trend and seasonal components with exogenous variables. 

Solution
The technical part of this assignment developed and implemented using python on a machine with specifications defined below:
- Programming language: Python 3.7.5
- IDE: PyCharm
- OS: Windows 10
- Processor: Intel® Core i7 CPU 2.20GHz 
- RAM: 8.00 GB
- System type: 64-bit Operating System, x64-based processor
- CPU Runtime:
* VARMAX model: 0.005 sec
* VAR model: 2.87 sec
* VARMA model: 0.005 sec

To address this problem, three classes have been defined following by a main.py file to run the whole project. These three classes consist of “preprocessing”, “models” and “forecast”. I tried to implement some aspects of OOP in this project such as encapsulation, polymorphism (overloading) and inheritance. Each of these will be explained on the next parts of the report.

The classes developed and inherits from each other as follow:
Forecast-> models-> Preprocessing

The “preprocessing” class is sub classed in “models” class and the “models” class is sub classed in “forecast” class. The classes functions are accessible by defining one object from “forecast” class. Also, some of the functions has defined privately in each class.

The path to the file would be defined in main.py file as follow:

```javascript
path = 'D:\\Audela Assignment\\dataset.csv'
ds = pd.read_csv(path, delimiter=';')  # read the dataset
```


By creating an object from “forecast” class, the methods from “preprocessing” class is called to perform data preparation phase.
```javascript
# create object 
obj = forecast(ds)

# dataPreprocessing method from preprocessing called to perform data preparation phase
obj.dataPreprocessing()

# plot data - from preprocessing class
obj.drawPlot(ds)
# ADFtest - from preprocessing class
obj.ADFtest()


```
 
By passing the dataset into preprocessing class, the unnecessary fields will be removed. Since the dataset doesn’t have any missing data, the methods related to enhance a dataset with missing data have not been implemented. Also, the data types will be converted to float64 and then will be normalize in case of using any Machine Learning models.
The preprocessing class has been defined as follow:
```javascript
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class preprocessing:
    def __init__(self, p):
        self.dataset = p
        self.target = pd.read_csv('D:\\Audela Assignment\\target.csv', delimiter=';')  # read the targets

    # data preparation phase
    def dataPreprocessing(self):
        del self.dataset['Index']
        del self.target['Index']

        for i in range(3):
            self.dataset['Val_' + str(i + 1)] = self.dataset['Val_' + str(i + 1)].astype(np.float64)
            self.target['Val_' + str(i + 1)] = self.target['Val_' + str(i + 1)].astype(np.float64)

    # normalize data, in case of using ML models
    def __norm(self, x, desc):
        return (x - desc.loc['mean']) / desc.loc['std']

    # get the statistical features of dataset
    def __getStats(self, data):
        stats = data.describe()
        return stats

    # visualize the data trend
    def drawPlot(self, data):
        print(data)
        data[['Val_1', 'Val_2', 'Val_3']].plot()
        plt.xlabel('Month')
        plt.show()

    # split the dataset following the form of 80(train) and 20(test)
    def __split(self):
        train = self.dataset[:int(.8 * (len(self.dataset)))]
        validation = self.dataset[int(.8 * (len(self.dataset))):]
```

 
To visualize the trend of features, the draw plot has been called with the dataset parameter:
 
Then the models are created via “models” class by calling the “createResult” method
```javascript
# VARMAX model
print('\n\n VARMAX model accuracy and result ****************************************************************')
obj.createResult(obj.select_model('VARMAX'), 0)

# VAR model
print('\n\n VAR model accuracy and result *******************************************************************')
obj.createResult(obj.select_model('VAR'), 0)

# VARMA model
print('\n\n VARMA model accuracy and result *****************************************************************')
obj.createResult(obj.select_model('VARMA'), 0)
```

 
By selecting VARMAX model from a simulated switch-case function from “models” class, the VARMAX model would be performed. 
```javascript
def select_model(self, i):
    switcher = {
        0: self.__VARMAXmodel(self.dataset, 5, 0),  # VARMAX model
        1: self.__VARmodel(),  # VAR model
        2: self.__VARMAXmodel(self.dataset, 1)  # VARMA model
    }
    return switcher.get(i, "Invalid key of model")
```
 
The VARMAX model has been defined to be performed both as VARMA and VARMAX model, which is based on the number of parameters it receives. Since the order of parameters in VARMA model is (1, 1), the only parameter is passed to this function is 1 for both p and q and when two parameters are passed, the VARMAX model performs.
```javascript
def __VARMAXmodel(self, data, *args):
    # set VARMA model parameters
    start_time = time.time()
    if len(args) == 1 and isinstance(args[0], int):
        p = args[0]
        q = args[0]
    # set VARMAx parameters
    elif len(args) == 2 and isinstance(args[1], int):
        p = args[0]
        q = args[1]

    # create model
    model = sm.tsa.VARMAX(data, order=(p, q), trend='nc')

    # fit the model
    model_result = model.fit(maxiter=100, disp=False)

    # print summary of model
    print(model_result.summary())

    # predicted values
    yhat = model_result.forecast(steps=4)

    print("--- %s seconds ---" % (time.time() - start_time))

    if p == q:
        pd.DataFrame(yhat).to_csv('VARMA_pred.csv', index=True)
    else:
        pd.DataFrame(yhat).to_csv('VARMAX_pred.csv', index=True)
    return yhat
```


 
Also, the VAR model defined as follow:
```javascript
def __VARmodel(self):
    start_time = time.time()
    # create model
    model = sm.tsa.VAR(self.dataset)

    # fit the model with lag order
    results = model.fit(maxlags=7, ic='aic')
    lag_order = results.k_ar

    # predict next 4 sequence
    yhat = results.forecast(self.dataset.values[-lag_order:], steps=4)
    # yhat = model_fit.forecast(model_fit.y, steps=4)

    print("--- %s seconds ---" % (time.time() - start_time))

    pd.DataFrame(yhat).to_csv('VAR_pred.csv', index=True)

    return pd.DataFrame(yhat)
```


 
Subsequently, creating the “forecast” class object at the beginning of “main”, the “forecast” would be performed to evaluate how correct the models work based on MAPE metrics. 
The “forecast” class defined as follow:
```javascript
from models import *
import numpy as np


class forecast(models):
    # initialize the primary variables in the instruction of class
    def __init__(self, p):
        super(forecast, self).__init__(p)
        self.normed_data = []
        self.normed_target = []

    # check whether or not they have unit roots by using the augmented Dickey Fuller (ADF) test
    def ADFtest(self):
        for i in range(3):
            print(sm.tsa.stattools.adfuller(self.dataset['Val_' + str(i + 1)]))

    # MAPE formulation
    def __MAPE(self, y, yhat):
        return np.mean(np.abs((y - yhat) / y)) * 100

    # calculate mape error for each vector
    def __calculateMAPE(self, y, yhat):
        # calculate mape
        res = []
        for i in range(4):
            res.append(self.__MAPE(np.array(y.iloc[i, :]),
                                 np.array(yhat.iloc[i, :])))
        return res

    # denormalize the predicted values (in case of using ML models)
    def __denormalization(self, yhat):
        return yhat * self.__getStats(self.dataset).loc['std'] + self.__getStats(self.dataset).loc['mean']

    # print the results using 3 models with MAPE error
    def createResult(self, yhat, flag):
        # calculate error and print the result
        res = self.__calculateMAPE(self.target, yhat)
        if flag == 1:
            # input is the normalized data
            print('MAPE Error for normalized data: {0:.2f} \n'.format(res), '\n')
            print('Converted normal value to real values: {0.2f} \n'.format(self.__denormalization(yhat)), '\n')

        else:
            # input is the main data
            res = self.__calculateMAPE(self.target, yhat)
            print('\n MAPE Error:')
            for i in range(4):
                print(i + self.dataset.shape[0] + 1, "{0:.2f}".format(res[i]),'%',)

            print('\n MAPE Error mean:')
            print("{0:.2f}".format(np.mean(res)),'%')
            print('Predicted values: \n', yhat)

```
 
Results
The predictions related to 3 models have been saved in 3 csv files. The MAPE metric for these models described as bellow:

Index | VARMAX | VAR | VARMA
------------ | ------------- | -------------| -------------
44 | 110.66 % | 149.46 % | 97.87 % 
45 | 32.11 % | 49.71 % | 39.02 %
46 | 28.63 % | 53.54 % | 35.31 % 
47 | 132.68 % | 147.88 % | 116.75 %

INDEX	VARMAX	VAR	VARMA
44	110.66 %	149.46 %	97.87 %
45	32.11 %	49.71 %	39.02 %
46	28.63 %	53.54 %	35.31 %
47	132.68 %	147.88 %	116.75 %



