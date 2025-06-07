from pandas_datareader.data import DataReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# labor = DataReader('AWHMAN', 'fred', start=start, end=end)                  # hours, manufacturing
# consumption = DataReader('USAPFCEQDSNAQ', 'fred', start=start, end=end)     # billions of dollars
# investment = DataReader('GPDI', 'fred', start=start, end=end)               # billions of dollars
# capital = DataReader('NFIRSAXDCUSQ', 'fred', start=start, end=end)          # million of dollars
# population = DataReader('CNP16OV', 'fred', start=start, end=end)            # thousands of persons
# recessions = DataReader('USRECQ', 'fred', start=start, end=end)

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


# solow residuals
alpha = 0.35

# Solow residual = log A_t
df['eA'] = df['y'] - alpha * df['k'] - (1 - alpha) * df['l']

# detrend
from statsmodels.tsa.filters.hp_filter import hpfilter

trend, cycle = {}, {}

for col in df.columns:
    cycle[col], trend[col] = hpfilter(df[col], lamb=1600)


cycle = pd.DataFrame(cycle)
trend = pd.DataFrame(trend)

# order data the same as .mod file
new_order = ['y', 'i', 'c', 'l', 'k','eA']
cycle_df = cycle[new_order]
cycle_df.to_csv('us_test.csv')

# ================================
# ===== PLOTTING =================
# ================================

cycle_cols = ['y', 'c', 'i', 'l']

fig, ax = plt.subplots(figsize=(8,6))

for col in cycle_cols:
    plt.plot(cycle_df[col], label=col.upper())


ax.fill_between(recessions.index, ylim[0]+1e-5, ylim[1]-1e-5, recessions,
                    facecolor='k', alpha=0.1)

plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.title('Cyclical Components of Y, C, I, and H (HP Filtered)')
plt.legend()
plt.tight_layout()
plt.show()
