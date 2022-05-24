import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

data = pd.read_csv('Automobile_data.csv')

data.head()

data.columns

data.shape

data.isnull().sum()

"""**There were no missing values present in the dataset**"""

data.dtypes

data.info()

"""**We can observe that those columns that have symbols are in object form as well as some columns should be of an integer type but are of an object type. Now let us detect which columns have symbols and if there are any other symbols too.**"""

for col in data.columns:
    print('{} : {}'.format(col,data[col].unique()))

"""**There are null values in our dataset in form of ‘?’ only but pandas are not reading them so we will replace them into np.nan form.**"""

data.head()

for col in data.columns:
    data[col].replace({'?':np.nan},inplace=True)

data.head()

"""**Now we convert the symbols ‘?’ into NaN format. Now let us check for missing values again.**"""

data.isnull().sum()

"""**Now we can see there were missing values in 7 columns in that we could see normalized-losses is having highest missing values**"""

sns.heatmap(data.isnull(),cbar=False,cmap='viridis')

"""As columns are having less missing values we can use any of the imputing techniques(Mean, Median (or) Mode)."""

data.columns

num_col = ['normalized-losses', 'bore',  'stroke', 'horsepower', 'peak-rpm','price']
for col in num_col:
    data[col]=pd.to_numeric(data[col])
    data[col].fillna(data[col].mean(), inplace=True)

data.head()

"""**Now we can see the missing values are replaced by using one of the statistical method mean.**"""

plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),cbar=True,annot=True,cmap='Blues')