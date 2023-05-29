import numpy as np
import pandas as pd
from WaveletDenoising import wavelet_noising
from GraphFeats import Graph
from StructureInFeats import Tensor, LBP
from HuMoments import hu_monments as Hu
from utils import TransferLabel, split, plot_confusion_matrix as pltcm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lightgbm import LGBMClassifier as LGBM
import warnings

warnings.filterwarnings('ignore')
mms = MinMaxScaler()


class loaddata:
    def __init__(self, name):
        self.name = name
        self.readpath = 'datasets/' + self.name + '.csv'
        self.rawData = pd.read_csv(self.readpath, encoding='gbk')
        self.labels = pd.DataFrame(self.rawData['Liths'], columns=['Liths'])
        self.depth = pd.DataFrame(self.rawData['Depths'], columns=['Depths'])
        self.data = wavelet_noising(self.rawData.iloc[:, 1:-1])

    def transfer(self):
        labels, Liths_map = TransferLabel(self.labels)

        return labels, Liths_map

    def classes(self):
        c = []
        if self.name == 'W1':
            # c = ['含介形虫泥岩', '含水泥质粉砂岩', '含水粉砂岩', '泥岩', '泥质粉砂岩', '粉砂岩', '粉砂质泥岩']
            c = ['BM', 'CS', 'WS', 'M', 'AS', 'S', 'SM']
        elif self.name == 'W2':
            # c = ['含水粉砂岩', '油页岩', '泥岩', '泥质粉砂岩', '粉砂岩', '粉砂质泥岩']
            c = ['WS', 'SH', 'M', 'AS', 'S', 'SM']
        elif self.name == 'W3':
            # c = ['含介形虫泥岩', '含介形虫粉砂质泥岩', '含水粉砂岩', '油迹粉砂岩', '油页岩', '泥岩', '泥质粉砂岩', '粉砂岩', '粉砂质泥岩']
            c = ['BM', 'OM', 'WS', 'OS', 'SH', 'M', 'AS', 'S', 'SM']
        elif self.name == 'W4':
            # c = ['含水粉砂岩', '泥岩', '泥质粉砂岩', '粉砂岩', '粉砂质泥岩']
            c = ['WS', 'M', 'AS', 'S', 'SM']
        elif self.name == 'W5':
            # c = ['油斑粉砂岩', '油迹粉砂岩', '泥岩', '泥质粉砂岩', '粉砂岩', '粉砂质泥岩']
            c = ['TS', 'OS', 'M', 'AS', 'S', 'SM']

        return c

    def InvaFeat(self):
        r1, r2, r3 = 10, 10, 10
        mlbp = LBP(self.data, r1)
        tens = Tensor(self.data, r2)
        hu = Hu(self.data, r3)
        return mlbp, tens, hu

    def Graph(self):
        g = Graph(self.data)
        X1, X2 = g.GFeats()
        return X1, X2


def LGBMrun(x_train, x_test, y_train, y_test, title=None):
    mms.fit(x_train)
    x_train = mms.transform(x_train)
    x_test = mms.transform(x_test)
    lgbm = LGBM(objective='multiclassova')
    lgbm.fit(x_train, y_train)
    ypre = lgbm.predict(x_test)
    return metrics(y_test, ypre, title=title)


def metrics(y_test, ypre, title=None):
    ACC = round(accuracy_score(y_test, ypre)*100, 2)
    F1 = round(f1_score(y_test, ypre, average='macro')*100, 2)
    print('ACC:{}'.format(ACC), ' F1:{}'.format(F1))

    cm = confusion_matrix(y_test, ypre)
    pltcm(cm, ld.classes(), title=title, normalize=True)  # normalize=True 绘制百分比

    return ACC, F1


def classify(x_train, x_test, y_train, y_test, FeatCom=None):
    lgbm_acc, lgbm_f1 = LGBMrun(x_train, x_test, y_train, y_test, title=FeatCom)
    return [lgbm_acc, lgbm_f1]


if __name__ == '__main__':

    res = pd.DataFrame(index=['W1', 'W2', 'W3', 'W4', 'W5'],
                       columns=['Original_ACC', 'Original_F1', 'Integrated_ACC', 'Integrated_F1'])

    files = ['W1', 'W2', 'W3', 'W4', 'W5']

    for name in files:
        print(name)
        ld = loaddata(name)
        y_train, y_test, train_ind, test_ind = split(ld.data, np.array(ld.labels['Liths']))

        mlbp, tens, hu = ld.InvaFeat()
        tmh = np.hstack((tens, mlbp, hu))
        X1, X2 = ld.Graph()
        X = np.hstack((X1, X2))

        #
        print('\nRaw')
        d = np.array(ld.rawData.iloc[:, 1:-1])
        x_train, x_test = d[train_ind], d[test_ind]
        classify(x_train, x_test, y_train, y_test, FeatCom=name + '_Raw')
        #

        print('\nOriginal features')
        d = np.array(ld.data)
        x_train, x_test = d[train_ind], d[test_ind]
        [a1, f1] = classify(x_train, x_test, y_train, y_test, FeatCom=name + '_Original_features')

        print('\nIntegrated features')
        xtmh = np.hstack((X, tmh))
        x_train, x_test = xtmh[train_ind], xtmh[test_ind]
        [a2, f2] = classify(x_train, x_test, y_train, y_test, FeatCom=name + '_Integrated_features')

        res.loc[name] = [a1, f1, a2, f2]

        print('\n', round(a2-a1, 2), round(f2-f1, 2))

        print(3 * '_______________________________________________\n')

    res.to_csv('results/results.csv')
