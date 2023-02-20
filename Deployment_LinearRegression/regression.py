import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/Data-Science-FMI/ml-from-scratch-2019/master/data/house_prices_train.csv"
df_train = pd.read_csv(url)
X = df_train[["OverallQual", "GrLivArea"]]
# print(X.dtypes)
X = (X - X.mean()) / X.std()
y = df_train[["SalePrice"]]
lr = LinearRegression()
lr.fit(X, y)

with open('./copied_to_docker/regression.pkl', 'wb') as model_pkl:
    pickle.dump(lr, model_pkl, protocol=2)

print("A trained multiple linear regression model has been built")
