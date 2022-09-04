'''
MIT License

Copyright (c) 2022 Alexander Yuryatin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from datetime import date

class SalesPlanSuccess:
    def __init__(self, data:pd.DataFrame, plan:int, sample_size:int = 50000):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The parameter 'data' of class SalesPlanSuccess can accept only pandas.DataFrame")
        if not isinstance(plan, int) and not isinstance(plan, float):
            raise TypeError("The parameter 'plan' of class SalesPlanSuccess can accept only regular Python integers and floats")
        if not isinstance(sample_size, int):
            raise TypeError("The parameter 'sample_size' of class SalesPlanSuccess can accept only regular Python integers")
        self.tt = data.copy()
        self.plan = plan
        self.sample_size = sample_size
        if (self.sample_size < 1000) or (self.sample_size > 10000000):
            raise ValueError("The parameter 'sample_size' of class SalesPlanSuccess must be between 1000 and 10 000 000")
        if self.plan <= 0:
            raise ValueError("The parameter 'plan' of class SalesPlanSuccess must be a positive number")
        if set(self.tt.columns) != set(['Year', 'Month', 'Sales']):
            raise ValueError("The parameter 'data' of class SalesPlanSuccess can only be pandas.DataFrame with three columns with names 'Year', 'Month', and 'Sales'")
        if self.tt.Year.dtypes != 'int64':
            raise ValueError("The column 'Year' in parameter 'data' of class SalesPlanSuccess must be of the dtype 'int64'")
        if self.tt.Month.dtypes != 'int64':
            raise ValueError("The column 'Month' in parameter 'data' of class SalesPlanSuccess must be of the dtype 'int64'")
        if self.tt.Sales.dtypes != 'int64' and self.tt.Sales.dtypes != 'float64':
            raise ValueError("The column 'Sales' in parameter 'data' of Class SalesPlanSuccess must be either of the dtype 'float64' or dtype 'int64'")
        if (self.tt.Year < 2000).sum() or (self.tt.Year > date.today().year).sum():
            raise ValueError("The column 'Year' in parameter 'data' of Class SalesPlanSuccess is expected to contain only the years between 2000 and the current year")
        if (self.tt.Month < 1).sum() or (self.tt.Month > 12).sum():
            raise ValueError("The column 'Month' in parameter 'data' of Class SalesPlanSuccess is expected to contain only the month numbers between 1 and 12")
        if (self.tt.Sales <= 0).sum():
            raise ValueError("The column 'Sales' in parameter 'data' of Class SalesPlanSuccess must contain only positive (non-zero) numbers")
        if self.tt.isnull().values.any():
            raise ValueError("The pandas.DataFrame in parameter 'data' of Class SalesPlanSuccess must not contain any empty values")
        self.tt['qEnd'] = 0
        self.tt.loc[np.int64(self.tt.Month.values) % 3 == 0, 'qEnd'] = 1
        self.finalMonth = self.tt.Month.iloc[-1]
        self.ytd_sales = self.tt.Sales.iloc[-self.finalMonth:].values.sum()
        self.tt['Sales'] = np.log(self.tt.Sales)
        self.finalSales = self.tt.Sales.iloc[-1]
        self.tt['Sales'] = self.tt.Sales.diff()
        self.tt.dropna(inplace=True)
        self.tt.reset_index(drop=True, inplace=True)
        self.model = sm.tsa.ARIMA(endog=self.tt.Sales.values, exog=self.tt.qEnd.values, order=(2, 0, 0))
        self.finalTwo = self.tt.Sales.iloc[-2:].values.reshape((2,1))
        
        
    def fit(self) -> None:
        self.model_fit = self.model.fit()
        self.qEnd = (((np.arange(int(self.tt.Month.iloc[-1])+1, 13) % 3) == 0) * self.model_fit.params[1]).reshape((5,1))
        self.ARs = self.model_fit.params[3:1:-1].reshape((1,2))
        self.finalMonth = int(self.tt.Month.iloc[-1])
        self.monthsToForecast = 12 - self.finalMonth
        self.m1 = np.dot(self.ARs, self.finalTwo)
        
    def simulate(self) -> None:
        if not hasattr(self, 'model_fit'):
            raise ValueError("Before calling method 'simulate' of an object of class SalesPlanSuccess you first have to fit the ARIMA model by calling method 'fit' of the same object")
        self.simul = np.random.normal(loc=self.model_fit.params[0], scale=np.sqrt(self.model_fit.params[4]), size=[self.monthsToForecast, self.sample_size])
        self.simul = self.qEnd + self.simul
        self.simul[0] = self.simul[0] + self.m1
        if self.monthsToForecast > 1:
            self.simul[1] = self.simul[1] + ((self.simul[0] * self.ARs[0,1]) + (self.finalTwo[1] * self.ARs[0,0]))
        if self.monthsToForecast > 2:
            for i in range(2, self.monthsToForecast):
                self.simul[i] = np.dot(self.ARs, self.simul[(i-2):i])
        self.finalDistibution = (np.exp(self.simul + self.finalSales)).sum(axis=0) + self.ytd_sales
        self.dfPlot = pd.DataFrame({'Sales': self.finalDistibution, 'Plan': 'Achieved'})
        self.dfPlot.loc[self.dfPlot.Sales < self.plan, 'Plan'] = 'Not achieved'
        
        self.left_x = min(self.finalDistibution)
        self.right_x = np.quantile(self.finalDistibution, 0.99)
        self.position = (self.plan - self.left_x) / (self.right_x - self.left_x)
        self.percent_not_achieved = 100*(self.finalDistibution < self.plan).mean()
        self.density = stats.kde.gaussian_kde(self.dfPlot.Sales)
        self.x1 = np.linspace(self.left_x, self.plan, 1000)
        self.x2 = np.linspace(self.plan, self.right_x, 1000)
        
    def plot(self) -> None:
        if not hasattr(self, 'model_fit'):
            raise ValueError("Before calling method 'plot' of an object of class SalesPlanSuccess you first have to fit the ARIMA model by calling method 'fit' and then conduct a simulation by calling method 'simulate' of the same object")
        if not hasattr(self, 'simul'):
            raise ValueError("Before calling method 'plot' of an object of class SalesPlanSuccess you first have to conduct a simulation by calling method 'simulate' of the same object")
        self.ax = plt.gca()
        plt.fill_between(self.x1, self.density(self.x1), color='orange')
        plt.fill_between(self.x2, self.density(self.x2), color='green')
        plt.text(self.position-0.01, 0.2, "Not achieved\n%.1f" % (self.percent_not_achieved,) + '%', color='black', fontsize = 15, transform=self.ax.transAxes, horizontalalignment='right')
        plt.text(self.position+0.01, 0.7, "Achieved\n%.1f" % (100 - self.percent_not_achieved,) + '%', color='black', fontsize = 15, transform=self.ax.transAxes)
        plt.xlim(self.left_x, self.right_x)
        plt.ylim(0)
        plt.box(False)
        plt.yticks([])
        plt.show()