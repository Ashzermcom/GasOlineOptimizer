import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from xgboost import XGBRegressor, plot_importance
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_regression

from model import ElasticNetModel, PCAModel, LassoModel, KernelPCAModel, RidgeModel, SimulatedAnnealing
from tools.train import xgb_parameters_search


# 导入数据并删除无意义的变量
def load_data(dir_path, drop_list=['id', '时间']):
    data = pd.read_excel(dir_path)
    for col in drop_list:
        data.drop(col, axis=1, inplace=True)
    return data


def nan_data_rate(df, n, ascending_=False, origin=True):
    """
    【Function】缺失率统计函数 nan_data_rate
    :param df: 需要处理的数据框
    :param n: 显示变量个数
    :param ascending_: 按缺失程度上升还是下降表示
    :param origin: 是否显示无缺失值失变量
    :return: 返回前n个缺失变量缺失率
    """
    if n > len(df.columns):  # 判断显示个数是否多于变量总数,如果超过则默认显示全部变量
        print('显示变量个数多于变量总数%i,将显示全部变量' % (len(df.columns)))
        n = len(df.columns)
    na_rate = df.isnull().sum() / len(df) * 100  # 统计各变量缺失率
    if origin:  # 判断为真则显示无缺失值的变量
        na_rate = na_rate.sort_values(ascending=ascending_)
        missing_data = pd.DataFrame({'Missing_Ratio': na_rate})
    else:  # 判断为负则显示只有缺失值的变量
        na_rate = na_rate.drop(na_rate[na_rate == 0].index).sort_values(ascending=ascending_)
        missing_data = pd.DataFrame({'Missing_Ratio': na_rate})
    return missing_data.head(n)


# 最小最大标准化
def min_max_normalize(df):
    """
        将数据框中所有变量 最小最大值标准化
    Args:
        df: 原始未标准化的数据框
    return: 标准化后的数据框
    """
    #
    for col in df.columns:
        min_val = np.min(df[col])
        max_val = np.max(df[col])
        df[col] = (df[col] - min_val) / (max_val - min_val)
        print("{} 变量已经被最小最大标准化".format(col))
    return df


# 数据预处理
def pre_process(df):
    df['RON_LOSS_RATE'] = df['RON_LOSS'] / (df['材料辛烷值RON'] + 1e-8)
    return df


# 方差筛选
def variance_select(df, k):
    """
        选取方差最大的前k个变量
    Args:
        df: 需要分析的数据框
        k: 选取变量的个数
    return: 变量列名
    """
    var = df.apply(lambda x: x.var())
    var = var.sort_values()
    return var.tail(k).index


# 特征提取
def feature_select(df):
    base = ['材料辛烷值RON', 'D121去稳定塔流量', '还原器温度', 'E-101D壳程出口管温度', 'D-204液位', 'D123冷凝水罐液位',
            'D-123压力', 'D-121水液位', 'D-102温度', '原料汽油硫含量',
            'TAG表和PID图未见PDI-2107点，是否为DI-2107', '稳定塔顶回流流量', '热循环气去R101底提升气管流量',
            '空气预热器空气出口温度', '低压热氮气压力', 'R-101下部床层压降', 'R-101床层中部温度', 'R-101床层下部温度',
            'P-101B入口过滤器差压', 'ME-109过滤器差压', 'ME-105过滤器压差', 'F-101辐射室出口压力',
            'S_ZORB AT-0004', 'S_ZORB AT-0011', 'D-201含硫污水液位', 'D101原料缓冲罐压力']
    # base = ['还原器温度', '预热器出口空气温度', '0.3MPa凝结水出装置流量',
    #         '对流室出口温度', 'E-101ABC壳程出口温度', 'E-101壳程出口总管温度.1', 'E-101DEF壳程出口温度',
    #         'E-101ABC管程出口温度', 'E-101DEF管程出口温度', '塔顶回流罐D201液位', '1.1步骤PIC2401B.OP',
    #         '1#催化汽油进装置流量', 'R-101床层中部温度.1', 'A-202A/B出口总管温度', 'D-102温度',
    #         'E-101A壳程出口管温度', 'D-125液位', '反应器质量空速', '8.0MPa氢气至反吹氢压缩机出口.1', '反应器藏量',
    #         '由PDI2104计算出', '反应器顶底压差', '8.0MPa氢气至循环氢压缩机入口.1',
    #         'EH-102加热元件/A束温度', 'K-102A进气压力', 'EH-102出口空气总管温度', 'S-ZORB.FT_1002.TOTAL',
    #         ]
    return df[base]


def extract_x(df, columns):
    x = df.copy()
    for col in columns:
        x.drop(col, axis=1, inplace=True)
    return x


def main_pca():
    df = pd.read_excel('/Users/ashzerm/item/GasOline/data/stand_oline.xlsx')
    df = min_max_normalize(df)
    y = df['RON_LOSS'].copy()
    x = extract_x(df, columns=['RON_LOSS', '饱和烃', '烯烃', '芳烃', '溴值', '密度', '产品硫含量', '产品辛烷值RON'])
    x = np.array(x)
    # pca = PCAModel(20)
    pca = KernelPCAModel(20, 'poly')
    x_pca = pca.train(x)
    # print(x_pca)
    xgb_parameters_search(x_pca, y)
    els = ElasticNetModel()
    els.grid_search_alpha_rho(x, y)
    els.test_elastic_net_alpha_rho(x, y)


def main_elastic_net():
    # df = pd.read_excel('/Users/ashzerm/item/GasOline/data/oline_xy.xlsx')
    df = pd.read_excel('/Users/ashzerm/item/GasOline/data/stand_oline.xlsx')
    df = min_max_normalize(df)
    y = df['RON_LOSS'].copy()
    # x = extract_x(df, columns=['RON_LOSS_RATE'])
    x = feature_select(df)
    els = ElasticNetModel()
    els.train(x, y)
    els.test(x, y)
    # els.grid_search_alpha_rho(x, y)
    els.test_elastic_net_alpha_rho(x, y)
    las = LassoModel(alpha=0.01)
    las.test_lasso_alpha(x, y)
    las.train(x, y, 5)
    # xgb_parameters_search(x, y)
    # xgb_r = XGBRegressor(max_depth=5, n_estimators=200, learning_rate=0.01)
    # xgb_r.fit(x, y)
    # plot_importance(xgb_r)
    # plt.show()
    # regr = RidgeModel()
    # regr.train(x, y)


# 自定义创建箱型图的函数
def create_box(data, x_name='', y_name='', title='', order_=[], scatter_=True):
    sns.boxplot(x=x_name, y=y_name, data=data,
                linewidth=2,  # 线宽
                width=0.8,  # 箱子之间的间隔比例
                fliersize=3,  # 异常点大小
                palette='hls',  # 设置调色板
                whis=1.5,  # 设置IQR
                notch=False,  # 设置是否以中值做凹槽
                order=order_)  # 筛选类别
    if scatter_:
        sns.swarmplot(x=x_name, y=y_name, data=data, color='k', size=3, alpha=0.6)
    plt.title(title)
    plt.show()
    return


def main_q3():
    df = pd.read_excel('/Users/ashzerm/item/GasOline/data/stand_oline.xlsx')
    # df = min_max_normalize(df)
    y = df['RON_LOSS'].copy()
    # x = extract_x(df, columns=['RON_LOSS_RATE'])
    x = feature_select(df)
    bins = [-0.1, 0.6, 1.0, 1.5, 2, 3]
    x['RON_LOSS_LEVEL'] = pd.cut(df['RON_LOSS'], bins=bins, labels=['0.6以下', '0.6-1.0', '1.0-1.5', '1.5-2', '2-3'])
    for col in df.columns:
        try:
            create_box(x, 'RON_LOSS_LEVEL', col, "{}与辛烷值损失RON箱型图".format(col),
                       ['0.6以下', '0.6-1.0', '1.0-1.5', '1.5-2', '2-3'])
        except:
            pass


def main_q5():
    df = pd.read_excel('/Users/ashzerm/item/GasOline/data/stand_oline.xlsx')
    # df = min_max_normalize(df)
    y = df['RON_LOSS'].copy()
    # x = extract_x(df, columns=['RON_LOSS_RATE'])
    x = feature_select(df)
    limits = pd.read_excel('/Users/ashzerm/item/GasOline/data/oline_limit.xlsx')
    model = RidgeModel()
    model.train(x, y)
    max_limit = np.array(limits.iloc[0, :])
    min_limit = np.array(limits.iloc[1, :])
    delta = np.array(limits.iloc[2, :])
    input_data = np.array(x.iloc[0, :])
    sa = SimulatedAnnealing(delta, max_limit, min_limit)
    sa.train(input_data, model.model, 0.1, 2, 20, y[0])


def plot_trace(ron=[1.4, 1.3553043829347597, 1.3671890877064197, 1.3243602007867263, 1.3179676106320226,
                    1.3088130218079908, 1.3161801589616564, 1.3035878489713886, 1.284602948826417,
                    1.2880308004261734, 1.2911156211639847, 1.3073715760770739, 1.1510164702696528,
                    1.1377735198593841, 1.131438297483916, 1.0976458271102327, 1.0934283202965585,
                    1.0868271489748729, 1.0780894004177775, 0.9183064430106143, 0.8739265498089788]
, s=[3.2, 3.1, 3.3, 3, 3.2,
     3.4, 3.1, 3, 3.2,
     3.1, 3.3, 3.3, 3,
     3.1, 3.1, 3.3, 3.2,
     3.1, 3.3, 3, 3.2]):
    # 绘制三维图
    x = list(range(1,22))
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot(x, s, ron)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.set_zlabel('辛烷损失RON')
    ax.set_title('变量操作优化')
    plt.show()


if __name__ == '__main__':
    # main_pca()
    # main_elastic_net()
    # main_q3()
    main_q5()
    # plot_trace()
