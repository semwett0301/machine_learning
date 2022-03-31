import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("fourth.csv").set_index("id")
predict_data = data.drop("Class", axis=1)
result_data = data["Class"]

classificator_1 = KNeighborsClassifier(n_neighbors=1, p=2).fit(predict_data, result_data)
print(classificator_1.kneighbors([[39, 22]], n_neighbors=1, return_distance=1))

classificator_2 = KNeighborsClassifier(n_neighbors=3, p=2).fit(predict_data, result_data)
print(classificator_2.kneighbors([[39, 22]], n_neighbors=3))

classificator_3 = KNeighborsClassifier(n_neighbors=3, p=2).fit(predict_data, result_data)
print(classificator_3.predict([[39, 22]]))

classificator_4 = KNeighborsClassifier(n_neighbors=1, p=1).fit(predict_data, result_data)
print(classificator_4.kneighbors([[39, 22]], n_neighbors=1, return_distance=1))

classificator_5 = KNeighborsClassifier(n_neighbors=1, p=1).fit(predict_data, result_data)
print(classificator_5.kneighbors([[39, 22]], n_neighbors=3))

classificator_6 = KNeighborsClassifier(n_neighbors=3, p=1).fit(predict_data, result_data)
print(classificator_6.predict([[39, 22]]))
