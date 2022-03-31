import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("six.csv").set_index('competitorname').drop(index=['Milky Way', 'One dime', 'Fun Dip']).reset_index()

orig_data = pd.read_csv("six.csv")

predict_data = data.drop(['competitorname', 'Y', 'winpercent'], axis=1)
result_data = data['Y']

log = LogisticRegression().fit(predict_data, result_data)

test_1_data = pd.read_csv("candy-test.csv").drop(
    np.where(pd.read_csv("candy-test.csv")['competitorname'] != 'Tootsie Roll Midgies')[0]).drop(
    ['competitorname', 'Y'], axis=1)
test_2_data = pd.read_csv("candy-test.csv").drop(
    np.where(pd.read_csv("candy-test.csv")['competitorname'] != 'Swedish Fish')[0]).drop(
    ['competitorname', 'Y'], axis=1)

print(log.predict_proba(test_1_data))
print(log.predict_proba(test_2_data))

test_data = pd.read_csv("candy-test.csv")
predict_data_test = test_data.drop(['competitorname', 'Y'], axis=1)

print(recall_score(test_data['Y'], log.predict(predict_data_test)))
print(precision_score(test_data['Y'], log.predict(predict_data_test)))

fpr, tpr, thresholds = roc_curve(test_data['Y'], log.predict(predict_data_test), pos_label=0)
print(auc(fpr, tpr))
