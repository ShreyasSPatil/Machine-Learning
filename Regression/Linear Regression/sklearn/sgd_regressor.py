import pandas as pd

daily_data = pd.read_csv("../datasets/Bike Sharing/day.csv")
hourly_data = pd.read_csv("../datasets/Bike Sharing/hour.csv")

print(list(daily_data.columns))