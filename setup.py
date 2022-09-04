from distutils.core import setup

setup(
    name = 'salesplansuccess',
    packages = ['salesplansuccess'],
    version = '0.0.3',
    license = 'MIT',
    description = 'This package helps estimate the probability of achieving the annual sales plan for a specific product based on historical monthly sales data with the help of ARIMA modeling and Monte Carlo simulation',
    url = 'https://github.com/yuryatin/salesplansuccess',
    download_url = 'https://github.com/yuryatin/salesplansuccess/archive/refs/tags/v0.0.3.tar.gz',
    keywords = ['sales', 'forecast', 'arima', 'simulation', 'pharma', 'medications', 'pharmacy', 'pharmaceutical', 'ics', 'offtake'],
    classifiers = [],
    install_requires = ['numpy','pandas','matplotlib','statsmodels', 'scipy', 'datetime']
)
