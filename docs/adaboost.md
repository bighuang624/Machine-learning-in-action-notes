## AdaBoost 元算法

### 原理

**元算法（meta-algorithm）**/**集成方法（ensemble method）**：对其他分类器进行组合

#### bagging：基于数据随机重抽样的分类器构建方法

**自举汇聚法（bootstrap aggregating）**，也称为 **bagging 方法**。在原数据集中随机放回抽样得到 S 个新数据集，将某个学习算法分别作用于每个数据集，就得到 S 个分类器。每次对新数据进行分类时，将 S 个分类器分类结果中最多的类别作为最后的分类结果。

#### boosting

**boosting（提升）** 中所使用的多个分类器的类型是一致的，但是每个新分类器都根据已训练出的分类器的性能来进行训练。和 bagging 不同，boosting 中分类器权重并不相等，代表其对应分类器在上一轮迭代中的成功度。

boosting 方法拥有多个版本，AdaBoost 是其中一个最流行的版本。

#### AdaBoost

AdaBoost 以弱学习器作为基分类器，并且输入数据，使其通过权重向量进行加权。在第一次迭代中，所有数据都等权重。但是在后续的迭代中，前次迭代中分错的样本的权重会增大，分类正确的样本权重减小。每个分类器的权重值 $\alpha$ 计算公式如下：

$$\alpha = \frac{1}{2}ln(\frac{1-\epsilon}{\epsilon})$$

其中，$\epsilon$ 为对应分类器的错误率。

权重向量 $D$ 的计算方法：

* 如果某个样本被正确分类，那么该样本的权重更改为：
  $$D\_i^{(t+1)} = \frac{D\_i^{(t)}e^{-\alpha}}{Sum(D)}$$
* 如果某个样本被错分，那么该样本的权重更改为：
  $$D\_i^{(t+1)} = \frac{D\_i^{(t)}e^{\alpha}}{Sum(D)}$$

AdaBoost 算法会不断地重复训练和调整权重的过程，直到训练误分类率为 0 或者弱分类器的数目达到用户的指定值为止。最终结果采取加权多数表决方法得到。

![AdaBoost.png](https://raw.githubusercontent.com/bighuang624/pic-repo/master/AdaBoost.png)

### 提升树

以决策树为基函数的提升方法称为**提升树（boosting tree）**。这里选择的弱分类器是**决策树桩（decision stump）**，它仅基于单个特征做二分的决策。

#### 代码实现

```py
# 通过阈值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

# 构建单层决策树
def buildStump(dataArr, classLabels, D):
    '''
    伪代码：
    将最小错误率 minError 设为正无穷
    对数据集中的每一个特征（第一层循环）：
        对每个步长（第二层循环）：
            对每个不等号（第三层循环）：
                建立一棵单层决策树并利用加权数据集对它进行测试
                如果错误率低于 minError，则将当前单层决策树设为最佳单层决策树
    返回最佳单层决策树
    '''
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # 计算加权错误率
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
        return bestStump, minError, bestClasEst

# 基于单层决策树的 AdaBoost 训练过程
def AdaBoostTrainDS(dataArr, classLabels, numIt=40):
    '''
    伪代码：
    对每次迭代：
        利用 buildStump() 函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树数组
        计算 alpha
        计算新的权重向量 D
        更新累计类别估计值
        如果错误率等于 0.0，则退出循环
    '''
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D: ", D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 用于确保没有错误时不会发生除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # 为下一次迭代计算 D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 错误率累加计算
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr

# AdaBoost 分类函数
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)
```

### 总结

AdaBoost 算法：

* 优点：泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整
* 缺点：对离群点敏感
* 适用数据类型：数值型和标称型

<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
   tex2jax: {inlineMath: [ ['$', '$'] ],
         displayMath: [ ['$$', '$$']]}
 });
</script>

<script src="https://cdn.bootcss.com/mathjax/2.7.4/latest.js?config=default"></script>