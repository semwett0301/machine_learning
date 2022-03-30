import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

data = pd.read_csv("candy-data.csv")


lin = LinearRegression().fit(data[['chocolate', 'fruity', 'caramel', 'peanutyalmondy']], data[['winpercent']])