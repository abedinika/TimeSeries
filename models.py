import statsmodels.api as sm
import time

from preprocessing import *


class models(preprocessing):
    def __init__(self, p):
        super(models, self).__init__(p)

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

    def select_model(self, i):
        switcher = {
            'VARMAX': self.__VARMAXmodel(self.dataset, 5, 0),  # VARMAX model
            'VAR': self.__VARmodel(),  # VAR model
            'VARMA': self.__VARMAXmodel(self.dataset, 1)  # VARMA model
        }
        return switcher.get(i, "Invalid key of model")

