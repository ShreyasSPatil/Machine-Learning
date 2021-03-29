import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

# Variables
data_directory = '../datasets/Wine Quality/'
paths = [data_directory + 'winequality-red.csv', data_directory + 'winequality-white.csv']


# Custom function for Data Importing
def read_wine_data(files):
    wine_features = list(map(lambda feature: feature.strip('"'), list(pd.read_csv(files[0]).columns)[0].split(';')))
    templist = []
    for file in files:
        data = pd.read_csv(file, skiprows=1)
        for index, row in data.iterrows():
            templist.append(list(map(lambda record: float(record), list(row[0].split(';')))))
    wine_data = pd.DataFrame(data=templist, columns=wine_features)
    return wine_data


# A generic function to split data into training features, training target, testing features and testing target
def split_data(data):
    train, test = train_test_split(data, test_size=0.2)
    print('Training Data Size: ' + str(train.shape))
    print('Testing Data Size: ' + str(test.shape))
    return train.iloc[:, :-1], train.iloc[:, [-1]], test.iloc[:, :-1], test.iloc[:, [-1]]


# Importing train and test datasets
winedata = read_wine_data(paths)

# Descriptive statistics
print(winedata.median())
print(winedata.describe())
sns.pairplot(winedata, hue='quality')
plt.show()

# Splitting data into X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = split_data(winedata)

# Perform linear regression using scikit-learn linear model
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(X_train, y_train)
pred = regr.predict(X_test)

# Print results
print('Coefficients: ' + str(regr.coef_))
print('Mean Squared Error: ' + str(metrics.mean_squared_error(y_test, pred)))
print('Mean Absolute Error: ' + str(metrics.mean_absolute_error(y_test, pred)))
