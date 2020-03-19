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
