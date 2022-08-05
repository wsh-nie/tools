import numpy as np
import numpy.random as random
from numpy.core.fromnumeric import *


class DBSCAN:
    def __init__(self, e=0.1, minPts=4):
        self.e = e
        self.minPts = minPts

    @staticmethod
    def calDist(X1, X2):
        # 计算两个向量之间的欧式距离
        dis = 0
        for x1, x2 in zip(X1, X2):
            dis += (x1 - x2) ** 2
        return dis ** 0.5

    def getNeibor(self, point, pointList):
        # 获取一个点的ε-邻域（记录的是索引）
        res = []
        for i in range(shape(pointList)[0]):
            if self.calDist(point, pointList[i]) < self.e:
                res.append(i)
        return res

    def pointsClust(self, pointList):
        coreObjs = {}   # 初始化核心对象集合
        C = {}
        n = len(pointList)
        # 找出所有核心对象，key是核心对象的index，value是ε-邻域中对象的index
        for i in range(n):
            neibor = self.getNeibor(pointList[i], pointList)
            if len(neibor) >= self.minPts:
                coreObjs[i] = neibor
        oldCoreObjs = coreObjs.copy()
        k = 0   # 初始化聚类簇数
        notAccess = list(range(n))  # 初始化未访问样本集合（索引）
        while len(coreObjs) > 0:
            OldNotAccess = []
            OldNotAccess.extend(notAccess)
            cores = coreObjs.keys()
            # 随机选取一个核心对象
            randNum = random.randint(0, len(cores))
            cores = list(cores)
            core = cores[randNum]
            queue = [core]
            notAccess.remove(core)
            while len(queue) > 0:
                q = queue[0]
                del queue[0]
                if q in oldCoreObjs.keys():
                    delte = [val for val in oldCoreObjs[q] if val in notAccess]     # Δ = N(q)∩Γ
                    queue.extend(delte)                                             # 将Δ中的样本加入队列Q
                    notAccess = [val for val in notAccess if val not in delte]      # Γ = Γ\Δ
            k += 1
            C[k] = [val for val in OldNotAccess if val not in notAccess]
            for x in C[k]:
                if x in coreObjs.keys():
                    del coreObjs[x]
        return C


def main():
    pointList = [[0.697, 0.46], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
                 [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
                 [0.639, 0.161], [0.657, 0.198], [0.36, 0.37], [0.593, 0.042], [0.719, 0.103], [0.359, 0.188],
                 [0.339, 0.241], [0.282, 0.257], [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437],
                 [0.525, 0.369], [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]]

    print(len(pointList))
    dbscan = DBSCAN(0.11, 5)
    C = dbscan.pointsClust(pointList)
    print(C)
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    import matplotlib.pyplot as plt
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for j in range(len(datas)):
            X.append(pointList[datas[j]][0])
            Y.append(pointList[datas[j]][1])
        plt.scatter(X, Y, marker='o', color=color[i % len(color)], label=i)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
