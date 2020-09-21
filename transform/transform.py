import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

data = pd.read_excel("/Users/ashzerm/item/GasOline/data/oline.xlsx")
target = np.array(data['RON_LOSS'].copy())
data = data[data.columns[16:]]
data = np.array(data)

estimator = XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5)

estimator.fit()
selector = RFE(estimator=estimator, n_features_to_select=10)
selector.fit(data, target)

print("N_features {}".format(selector.n_features_))
print("Support is {}".format(selector.support_))
print("Ranking is {}".format(selector.ranking_))

['K-103A排气压力', '反吹气压力', 'K-103A进气压力', 'K-101A排气压力', 'K-101A进气压力', 'D-110底压力', '反应过滤器压差.1',
 '热氮气过滤器ME-113差压', '稳定塔液位', '塔顶回流罐D201液位', '原料缓冲罐液位', 'K-103A排气温度', 'K-101A进气温度', 'K-103A进气温度',
 '紧急氢气总管', '1.1步骤PIC2401B.OP', 'S-ZORB.FT_1002.TOTAL', 'S-ZORB.FT_1204.TOTAL', 'S-ZORB.FT_1202.TOTAL', 'EH101出口',
 'S_ZORB AT-0008', 'S_ZORB AT-0011', 'D-109松动风流量', '原料辛烷值RON']
