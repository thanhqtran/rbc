from pandas_datareader.data import DataReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the datasets from FRED
start = '1970-01-01'
# end = '2014-12-01'
end = '2024-06-01'
indprod = DataReader('IPMAN', 'fred', start=start, end=end)
income = DataReader('W875RX1', 'fred', start=start, end=end)
# sales = DataReader('CMRMTSPL', 'fred', start=start, end=end)
emp = DataReader('PAYEMS', 'fred', start=start, end=end)
# dta = pd.concat((indprod, income, sales, emp), axis=1)
# dta.columns = ['indprod', 'income', 'sales', 'emp']

HMRMT = DataReader('HMRMT', 'fred', start='1967-01-01', end=end)
CMRMT = DataReader('CMRMT', 'fred', start='1997-01-01', end=end)
HMRMT_growth = HMRMT.diff() / HMRMT.shift()
sales = pd.Series(np.zeros(emp.shape[0]), index=emp.index)

# Fill in the recent entries (1997 onwards)
sales[CMRMT.index] = CMRMT['CMRMT']

# Backfill the previous entries (pre 1997)
idx = sales.loc[:'1997-01-01'].index
for t in range(len(idx)-1, 0, -1):
    month = idx[t]
    prev_month = idx[t-1]
    sales.loc[prev_month] = sales.loc[month] / (1 + HMRMT_growth.loc[prev_month].values)
dta = pd.concat((indprod, income, sales, emp), axis=1)
dta.columns = ['indprod', 'income', 'sales', 'emp']
dta.loc[:, 'indprod':'emp'].plot(subplots=True, layout=(2, 2), figsize=(15, 6));

# Get some data
start='1975-01'
end = '2023-03'

labor = DataReader('HOHWMN02USQ065S', 'fred', start=start, end=end)         # hours
consumption = DataReader('PCECC96', 'fred', start=start, end=end)           # billions of dollars
investment = DataReader('GPDI', 'fred', start=start, end=end)               # billions of dollars
capital = DataReader('NFIRSAXDCUSQ', 'fred', start=start, end=end)          # million of dollars
population = DataReader('CNP16OV', 'fred', start=start, end=end)            # thousands of persons
recessions = DataReader('USRECQ', 'fred', start=start, end=end)
recessions = recessions.resample('QS').last()['USRECQ'].iloc[1:]

# Collect the raw values
raw = pd.concat((labor, consumption, investment, capital, population.resample('QS').mean()), axis=1)
raw.columns = ['labor', 'consumption', 'investment','capital', 'population']
raw['output'] = raw['consumption'] + raw['investment']

# convert data to units and normalize with population
y = (raw['output'] * 1e9) / (raw['population'] * 1e3)
i = (raw['investment'] * 1e9) / (raw['population'] * 1e3)
c = (raw['consumption'] * 1e9) / (raw['population'] * 1e3)
k = (raw['capital'] * 1e12)/(raw['population']*1e3)
h = raw['labor']      

# assemble into 1 dataset
dta = pd.DataFrame({
    'y': y,
    'i': i, 
    'c': c,
    'h': h,
    'k': k
})

# seasonally adjust the data
from statsmodels.tsa.seasonal import seasonal_decompose

def seasonally_adjust(series):
    result = seasonal_decompose(series.dropna(), model='multiplicative', period=4)
    return series / result.seasonal

dta_sa = dta.apply(seasonally_adjust)

# take logs of selected columns only
dta_log = dta_sa.copy()
dta_log['y'] = np.log(dta_sa['y'])
dta_log['i'] = np.log(dta_sa['i'])
dta_log['c'] = np.log(dta_sa['c'])
dta_log['l'] = np.log(dta_sa['h'])
dta_log['k'] = np.log(dta_sa['k'])
df = dta_log.copy()


# detrend with HP-filter
from statsmodels.tsa.filters.hp_filter import hpfilter

trend, cycle = {}, {}

for col in df.columns:
    cycle[col], trend[col] = hpfilter(df[col], lamb=1600)


cycle = pd.DataFrame(cycle)
trend = pd.DataFrame(trend)

new_order = ['y', 'i', 'c', 'l', 'k']
cycle_df = cycle[new_order]

cycle_df.to_csv('us_test.csv')
