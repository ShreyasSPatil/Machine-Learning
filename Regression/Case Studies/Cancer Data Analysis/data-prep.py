# Importing datasets wrangling libraries
import numpy as np
import pandas as pd

incd_data = pd.read_csv('data/Cancer/incd.csv', usecols=['State', 'FIPS', 'Age-Adjusted Incidence Rate([rate note]) - cases per 100,000', 'Average Annual Count', 'Recent Trend'])
print(incd_data.columns)
