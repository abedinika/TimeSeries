 



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

'''
path = 'D:\\Audela Assignment\\dataset.csv'
ds = pd.read_csv(path, delimiter=';')  # read the dataset
'''
 
By creating an object from “forecast” class, the methods from “preprocessing” class is called to perform data preparation phase.
 
By passing the dataset into preprocessing class, the unnecessary fields will be removed. Since the dataset doesn’t have any missing data, the methods related to enhance a dataset with missing data have not been implemented. Also, the data types will be converted to float64 and then will be normalize in case of using any Machine Learning models.
The preprocessing class has been defined as follow:
 
To visualize the trend of features, the draw plot has been called with the dataset parameter:
 
Then the models are created via “models” class by calling the “createResult” method.
 
By selecting VARMAX model from a simulated switch-case function from “models” class, the VARMAX model would be performed. 
 
The VARMAX model has been defined to be performed both as VARMA and VARMAX model, which is based on the number of parameters it receives. Since the order of parameters in VARMA model is (1, 1), the only parameter is passed to this function is 1 for both p and q and when two parameters are passed, the VARMAX model performs.
 
Also, the VAR model defined as follow:
 
Subsequently, creating the “forecast” class object at the beginning of “main”, the “forecast” would be performed to evaluate how correct the models work based on MAPE metrics. 
The “forecast” class defined as follow:
 
Results
The predictions related to 3 models have been saved in 3 csv files. The MAPE metric for these models described as bellow:
INDEX	VARMAX	VAR	VARMA
44	110.66 %	149.46 %	97.87 %
45	32.11 %	49.71 %	39.02 %
46	28.63 %	53.54 %	35.31 %
47	132.68 %	147.88 %	116.75 %



