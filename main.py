import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("candy-data.csv").set_index('competitorname').drop(index=['Milky Way', 'Mr Good Bar']).reset_index()

predict_data = data.drop(['competitorname', 'Y', 'winpercent'], axis=1)
result_data = data['winpercent']

lin = LinearRegression().fit(predict_data, result_data)

test_1_data = pd.read_csv("candy-data.csv").drop(
    np.where(pd.read_csv("candy-data.csv")['competitorname'] != 'Milky Way')[0]).drop(
    ['competitorname', 'Y', 'winpercent'], axis=1)
test_2_data = pd.read_csv("candy-data.csv").drop(
    np.where(pd.read_csv("candy-data.csv")['competitorname'] != 'Mr Good Bar')[0]).drop(
    ['competitorname', 'Y', 'winpercent'], axis=1)

print(lin.predict(test_1_data))
print(lin.predict(test_2_data))
print(lin.predict([[0, 1, 0, 1, 0, 1, 1, 0, 0, 0.93, 0.635]]))
