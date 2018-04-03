from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    """数据载入函数"""
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    """标准回归函数"""
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse!")
        return
    ws = xTx.I * (xMat.T * yMat)
    #ws = linalg.solve(xTx, xMat.T*yMat) ##########用于解决未知矩阵（可以不用考虑矩阵是否奇异）
    return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse!")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws ###########返回预测值yHat(i)

def testlwlr(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat ##################返回所有预测值组成的数组


# dataSet, labelSet = loadDataSet('ex0.txt')
# ws = standRegres(dataSet, labelSet)
#
# xMat = mat(dataSet)
# yMat = mat(labelSet)
# yHat = xMat * ws

# print(corrcoef(yHat.T, yMat))
# #绘制散点图和拟合直线
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
#
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy * ws
# ax.plot(xCopy[:,1], yHat)
# plt.show()

############局部加权线性回归预测测试
# dataSet, labelSet = loadDataSet('ex0.txt')
# xMat = mat(dataSet)
# yMat = mat(labelSet)
# print(yMat)
# yHat_1 = testlwlr(dataSet, dataSet, labelSet, 0.01)
# print(yHat_1)
# print(corrcoef(yHat_1, yMat))
# srtInd = xMat[:, 1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:, 1], yHat_1[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0], yMat.T.flatten().A[0],s=2,c='red')
# plt.show()


###################岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This is a singule, connt do inverse!")
        return
    ws = denom.I * xMat.T * yMat
    return ws

#############先要对特征进行标准化处理：减去均值，除以方差
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMeans = mean(yMat, 0)
    yMat = yMat - yMeans
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPt = 30
    wMat = zeros((numTestPt, shape(xMat)[1]))
    for i in range(numTestPt):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat



############应用回归预测鲍鱼年龄
def rssError(yArr, yHatArr):
    """均方误差和,用于衡量误差的大小"""
    return ((yArr - yHatArr) ** 2).sum()


####################前向逐步回归
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print (ws.T)
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat
#####testing code

abX, abY = loadDataSet("abalone.txt")
WMat = stageWise(abX, abY, 0.01, 200)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(WMat)
plt.legend("up right")
plt.show()




# ########不同平滑参数k下的误差大小
# abX, abY = loadDataSet('abalone.txt')
# print(shape(abX))
# yHat01 = testlwlr(abX[:99],abX[:99],abY[:99],0.1)
# yHat1 = testlwlr(abX[:99],abX[:99],abY[:99],1)
# yHat10 = testlwlr(abX[:99],abX[:99],abY[:99],10)
# print(rssError(abY[:99], yHat01))######## 56.8#########发现，核越小（k值越小），误差越小，但不能再数据集上
# print(rssError(abY[:99], yHat1))######## 429.9######都使用最小的核，因为使用最小的核将造成过拟合
# print(rssError(abY[:99], yHat10))####### 549.1#####从而导致训练效果好，但预测效果一般（即：泛化能力弱）
#
# ####################################
# ridgeWeights = ridgeTest(abX, abY)
# print(ridgeWeights)
#
# #######作图观察回归系数的变化
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()
