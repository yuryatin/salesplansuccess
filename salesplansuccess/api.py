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

class SalesPlanSuccess:
    def __init__(self, data:pd.DataFrame, plan:int, product:str = ''):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The parameter 'data' of class SalesPlanSuccess can accept only pandas.DataFrame")
        if not isinstance(plan, int) and not isinstance(plan, float):
            raise TypeError("The parameter 'plan' of class SalesPlanSuccess can accept only regular Python integers and floats")
        self.tt = data.copy()
        self.plan = plan
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
        self.tt.sort_values(by=['Year','Month'], inplace=True)
        self.tt.reset_index(drop=True, inplace=True)
        if self.tt.shape[0] < 7:
            raise Error("The pandas.DataFrame in parameter 'data' of Class SalesPlanSuccess must contain at least 7 months of observations for modeling")
        if self.tt[['Year','Month']].duplicated().any():
            raise ValueError("The pandas.DataFrame in parameter 'data' of Class SalesPlanSuccess must not contain duplicate Year/Month pairs")
        if not isinstance(product, str):
            raise TypeError("The parameter 'product' of class SalesPlanSuccess can accept only strings")
        self.product = 'of ' + product.capitalize() if product != '' else ''
        self.tt2 = self.tt.copy()
        self.tt2['SOY'] = False
        self.tt2.loc[self.tt2.Month == 1, 'SOY'] = True
        self.tt2[['Year', 'Month']] = self.tt2[['Year', 'Month']].diff()
        self.tt2.dropna(inplace=True)
        if not ((self.tt2.loc[self.tt2.SOY, 'Year'] == 1.0).all() and
                (self.tt2.loc[~self.tt2.SOY, 'Year'] == 0.0).all() and
                (self.tt2.loc[self.tt2.SOY, 'Month'] == -11.0).all() and
                (self.tt2.loc[~self.tt2.SOY, 'Month'] == 1.0).all()):
            raise ValueError("The pandas.DataFrame in parameter 'data' of Class SalesPlanSuccess must contain consecutive Year/Month rows without omiting individual months")
        del self.tt2
        self.tt['qEnd'] = 0
        self.tt.loc[np.int64(self.tt.Month.values) % 3 == 0, 'qEnd'] = 1
        self.finalMonth = self.tt.Month.iloc[-1]
        self.monthsToForecast = 12 - self.finalMonth if self.finalMonth < 12 else 12
        self.yearForcasted = self.tt.Year.iloc[-1] if self.finalMonth < 12 else self.tt.Year.iloc[-1] + 1
        self.ytd_sales = self.tt.Sales.iloc[-self.finalMonth:].values.sum() if self.finalMonth < 12 else 0.0
        self.tt['Sales'] = np.log(self.tt.Sales)
        self.finalSales = self.tt.Sales.iloc[-1]
        self.tt['Sales'] = self.tt.Sales.diff()
        self.tt.dropna(inplace=True)
        self.tt.reset_index(drop=True, inplace=True)
        self.model = sm.tsa.ARIMA(endog=self.tt.Sales.values, exog=self.tt.qEnd.values, order=(2, 0, 0))
        self.finalTwo = self.tt.Sales.iloc[-2:].values.reshape((2,1))
        self.startMonth = self.finalMonth + 1 if self.finalMonth < 12 else 1

        
    def fit(self, mode:str = "LSE") -> None:
        if not isinstance(mode, str):
            raise TypeError("The parameter 'mode' of method fit() of class SalesPlanSuccess can accept only strings")
        if (mode != 'LSE') and (mode != 'ARIMA'):
            raise ValueError("The parameter 'mode' of method fit() of class SalesPlanSuccess is expected to get either value 'LSE' (default) or 'ARIMA'")
        self.mode = mode
        if self.mode == 'LSE':
            self._fit_lse()
        else:
            self._fit_arima()
        self.params = pd.Series(data=np.concatenate((self.params, np.sqrt(self.params[-1]).reshape((1,)))), index=['mu', 'EoQ', 'AR1', 'AR2', 'var', 'sigma'])

            
    def _fit_arima(self) -> None:
        self.model_fit = self.model.fit()
        self.params = None
        self.params = self.model_fit.params
       
        
    def _fit_lse(self) -> None:
        self.y = self.tt.Sales.iloc[2:].values.reshape((self.tt.shape[0]-2, 1))
        self.x = self.tt[['Sales', 'qEnd']].copy()
        self.x['AR1'] = self.x.Sales.shift(1)
        self.x['AR2'] = self.x.Sales.shift(2)
        self.x = np.hstack([np.ones(self.tt.shape[0]-2).reshape((self.tt.shape[0]-2,1)), self.x[['qEnd', 'AR1', 'AR2']].iloc[2:].values])
        self.model_fit2 = np.linalg.lstsq(self.x, self.y, rcond=None)
        self.params = None
        self.params = np.concatenate((self.model_fit2[0], (self.y - np.dot(self.x, self.model_fit2[0])).var().reshape((1,1)))).reshape((5,))


    def summary(self) -> None:
        if not hasattr(self, 'params'):
            raise ValueError("Before calling method summary() of an object of class SalesPlanSuccess you first have to fit the ARIMA model by calling method 'fit' of the same object")
        print('\t  Coefficient estimates in %s\nMonthly drift:\t\t\t\t%6.3f\nEnd of quarter:\t\t\t\t%6.3f\nAR1:\t\t\t\t\t%6.3f\nAR2:\t\t\t\t\t%6.3f\nStandard deviation of residuals:\t%6.3f' % (self.mode, self.params['mu'], self.params['EoQ'], self.params['AR1'], self.params['AR2'], self.params['sigma']))
        
        
    def simulate(self, sample_size:int = 50000, mu:float = None, sigma:float = None, EoQ:float = None, AR1:float = None, AR2:float = None) -> None:
        if not isinstance(sample_size, int):
            raise TypeError("The parameter 'sample_size' of class SalesPlanSuccess can accept only regular Python integers")
        if not isinstance(mu, float) and not isinstance(mu, int) and mu is not None:
            raise TypeError("The parameter 'mu' of class SalesPlanSuccess can accept only regular Python floats and integers")
        if not isinstance(sigma, float) and not isinstance(sigma, int) and sigma is not None:
            raise TypeError("The parameter 'sigma' of class SalesPlanSuccess can accept only regular Python floats and integers")
        if not isinstance(EoQ, float) and not isinstance(EoQ, int) and EoQ is not None:
            raise TypeError("The parameter 'EoQ' of class SalesPlanSuccess can accept only regular Python floats and integers")
        if not isinstance(AR1, float) and not isinstance(AR1, int) and AR1 is not None:
            raise TypeError("The parameter 'AR1' of class SalesPlanSuccess can accept only regular Python floats and integers")
        if not isinstance(AR2, float) and not isinstance(AR2, int) and AR2 is not None:
            raise TypeError("The parameter 'AR2' of class SalesPlanSuccess can accept only regular Python floats and integers")
        if (sample_size < 1000) or (sample_size > 10000000):
            raise ValueError("The parameter 'sample_size' of method simulate() in class SalesPlanSuccess must be between 1000 and 10 000 000")
        if not hasattr(self, 'params'):
            raise ValueError("Before calling method 'simulate' of an object of class SalesPlanSuccess you first have to fit the ARIMA model by calling method 'fit' of the same object")
        if sigma is not None:
            if (sigma <= 0):
                raise ValueError("The parameter 'sigma' of method simulate() in class SalesPlanSuccess must be a positive (non-zero) number")
            self.params['sigma'] = sigma
            self.params['var'] = np.power(sigma, 2)
        if AR1 is not None:
            if (AR1 <= -1.0) or (AR1 >= 1.0):
                raise ValueError("The parameter 'AR1' of method simulate() in class SalesPlanSuccess must be between -1.0 and 1.0")
            self.params['AR1'] = AR1
        if AR2 is not None:
            if (AR2 <= -1.0) or (AR2 >= 1.0):
                raise ValueError("The parameter 'AR2' of method simulate() in class SalesPlanSuccess must be between -1.0 and 1.0")
            self.params['AR2'] = AR2
        if mu is not None:
            self.params['mu'] = mu
        if EoQ is not None:
            self.params['EoQ'] = EoQ
        self.qEnd = (((np.arange(self.startMonth, 13) % 3) == 0) * self.params['EoQ']).reshape((self.monthsToForecast,1))
        self.ARs = self.params[3:1:-1].values.reshape((1,2))
        self.m1 = np.dot(self.ARs, self.finalTwo)
        self.sample_size = sample_size
        self.simul = np.random.normal(loc=self.params['mu'], scale=self.params['sigma'], size=[self.monthsToForecast, self.sample_size])
        self.simul = self.qEnd + self.simul
        self.simul[0] = self.simul[0] + self.m1
        if self.monthsToForecast > 1:
            self.simul[1] = self.simul[1] + ((self.simul[0] * self.params['AR1']) + (self.finalTwo[1] * self.params['AR2']))
        if self.monthsToForecast > 2:
            for i in range(2, self.monthsToForecast):
                self.simul[i] = self.simul[i] + np.dot(self.ARs, self.simul[(i-2):i])
        self.finalDistibution = (np.exp(self.simul.cumsum(axis=0) + self.finalSales)).sum(axis=0) + self.ytd_sales       
        self.left_x = min(self.finalDistibution)
        self.left_margin = self.left_x if self.left_x < self.plan else self.plan
        self.right_x = np.quantile(self.finalDistibution, 0.99)
        self.right_margin = self.right_x if self.right_x > self.plan else self.plan
        self.position_plan = (self.plan - self.left_margin) / (self.right_margin - self.left_margin)
        self.percent_not_achieved = 100*(self.finalDistibution < self.plan).mean()
        self.density = stats.kde.gaussian_kde(self.finalDistibution)
        self.x1 = np.linspace(self.left_x, self.plan, 1000)
        self.x2 = np.linspace(self.plan, self.right_x, 1000)
        self.y1 = self.density(self.x1)
        self.y2 = self.density(self.x2)
        self.mode1 = np.argmax(self.y1)
        self.mode2 = np.argmax(self.y2)
        if max(self.y1) > max(self.y2):
            self.moda = self.x1[self.mode1]
        else:
            self.moda = self.x2[self.mode2]
        self.position_mode = (self.moda - self.left_margin) / (self.right_margin - self.left_margin)
        if self.position_plan > self.position_mode:
            self.vertical_position1 = 0.2
            self.vertical_position2 = 0.7
        else:
            self.vertical_position1 = 0.7
            self.vertical_position2 = 0.2
        self.summary()
        
        
    def plot(self, failure_color:str = 'orange', success_color:str = 'green') -> None:
        if not hasattr(self, 'params'):
            raise ValueError("Before calling method 'plot' of an object of class SalesPlanSuccess you first have to fit the ARIMA model by calling method 'fit' and then conduct a simulation by calling method 'simulate' of the same object")
        if not hasattr(self, 'simul'):
            raise ValueError("Before calling method 'plot' of an object of class SalesPlanSuccess you first have to conduct a simulation by calling method 'simulate' of the same object")
        if not isinstance(failure_color, str):
            raise TypeError("The parameter 'failure_color' of method plot() in class SalesPlanSuccess must be string")
        if not isinstance(success_color, str):
            raise TypeError("The parameter 'success_color' of method plot() in class SalesPlanSuccess must be string")
        if not is_color_like(failure_color):
            raise ValueError("The parameter 'failure_color' of method plot() in class SalesPlanSuccess must contain valid color")
        if not is_color_like(success_color):
            raise ValueError("The parameter 'success_color' of method plot() in class SalesPlanSuccess must contain valid color")
        self.ax = plt.gca()
        plt.fill_between(self.x1, self.y1, color = failure_color)
        plt.fill_between(self.x2, self.y2, color = success_color)
        plt.text(self.position_plan-0.01, self.vertical_position1, "Not achieved\n%.1f" % (self.percent_not_achieved,) + '%', color='black', fontsize = 15, transform=self.ax.transAxes, horizontalalignment='right')
        plt.text(self.position_plan+0.01, self.vertical_position2, "Achieved\n%.1f" % (100 - self.percent_not_achieved,) + '%', color='black', fontsize = 15, transform=self.ax.transAxes)
        plt.xlabel("Expected annual sales vs. the plan")
        plt.title("Expected annual sales %s in %4d" % (self.product, self.yearForcasted))
        plt.xlim(self.left_margin, self.right_margin)
        plt.ylim(0)
        plt.box(False)
        plt.yticks([])
        plt.show()