# Python package _salesplansuccess_
This Python package helps estimate the probability of achieving the annual sales plan in the middle of the year for the current year or at the end of the year for the next year for a specific product based on its historical monthly sales data with the help of ARIMA modeling and Monte Carlo simulation.

# Installation
This Python package is uploaded to the PyPI repository and therefore can be installed with `pip install salesplansuccess` command line instruction and updated (this is critical at this early stage of development) with `pip install salesplansuccess --upgrade`.

# Historical data format to feed
The only class `SalesPlanSuccess()` of this package accepts for its data parameter only *pandas.DataFrame* with three columns with the names *Year*, *Month*, and *Sales*. The columns *Year* and *Month* must be of the dtype `numpy.int64` and the column *Sales* must be either of the dtype `numpy.int64` or `numpy.float64`. All monthly sales data must be consecutive and positive (must not be zeros or omitted).

# Assumptions
The model behind the forecast assumes that the monthly sales changes' residuals are lognormally distributed and the logarithmic monthly sales time series is subject to an ARIMA(2,1,0) process with one external regressor, which is the end of a quarter (March, June, September or December). This lognormal assumption cannot accomodate the historical sales data with 'no sales' months. So, please, do not use this package for historical sales time series with zeros.<br/> This model also apparently assumes that the sales dynamics was subject to the same non-changing process in the past and will continue to follow it in the future (no new promo interventions are assumed).

# How to use it
Very easy. The package has only one class `SalesPlanSuccess()`, which is initiated with the historical sales pandas DataFrame, the current annual plan number, and optionally with the product name (for the plot). This class has only 4 'public' methods: `.fit()`, `.summary()`, `.simulate()`, `.plot()`, which are expected to be called in this order.<br/> The illustration of how to use this package can be found in the attached file [salesplansuccess_illustration.ipybn](https://github.com/yuryatin/salesplansuccess/blob/main/salesplansuccess_illustration.ipynb) .<br/> Briefly this can expressed as this:
```python
import pandas as pd
from salesplansuccess.api import SalesPlanSuccess
myHistoricalData = pd.read_excel('myfile.xlsx')
myAnnualPlan = 1000
sps = SalesPlanSuccess(data=myHistoricalData, plan=myAnnualPlan)
sps.fit()
sps.simulate()
sps.plot()
```
