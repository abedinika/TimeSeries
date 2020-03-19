from forecast import *
import pandas as pd
import numpy as np


np.set_printoptions(precision=2)

path = 'D:\\Audela Assignment\\dataset.csv'
ds = pd.read_csv(path, delimiter=';')  # read the dataset

# create object
obj = forecast(ds)

# dataPreprocessing method from preprocessing called to perform data preparation phase
obj.dataPreprocessing()

# plot data - from preprocessing class
obj.drawPlot(ds)
# ADFtest - from preprocessing class
obj.ADFtest()

# VARMAX model
print('\n\n VARMAX model accuracy and result ****************************************************************')
obj.createResult(obj.select_model('VARMAX'), 0)

# VAR model
print('\n\n VAR model accuracy and result *******************************************************************')
obj.createResult(obj.select_model('VAR'), 0)

# VARMA model
print('\n\n VARMA model accuracy and result *****************************************************************')
obj.createResult(obj.select_model('VARMA'), 0)


