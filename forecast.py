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
