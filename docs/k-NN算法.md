## k-近邻算法（kNN）

### 原理

输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后选择 k 个最相似数据（最近邻）中出现次数最多的分类标签。

### 实现思路

对未知类别属性的数据集中的每个点依次执行以下操作：

1. 计算已知类别数据集中的点与当前点之间的欧式距离；
2. 按照距离递增次序排序；
3. 选取与当前点距离最小的 k 个点；
4. 确定前 k 个点所在类别的出现频率；
5. 返回前 k 个点所出现频率最高的类别作为当前点的预测分类。

### 三要素

对于固定的训练集，只要这三点确定了，算法的预测方式也就确定了：

* **k 值的选取**：k 值较小则减小训练误差，增大泛化误差；k 值较大则增加训练误差，减小泛化误差。
* **距离的度量方式**：一般用欧式距离，也可以使用曼哈顿距离。
* **分类决策规则**：一般采用多数表决法。

### 代码实现

```py
def classify0(inX, dataSet, labels, k):
    '''
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 将 classCount 字典分解为元组列表，然后用 itemgetter 方法，按照第二个元素的次序对元组进行排序（逆序）
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

### kd 树

暴力实现 k-NN 算法不适用于样本量或者样本特征数较大的情况，因为算法的时间效率较低。用 kd 树实现可以提高效率。

### 总结

* 优点：精度高、对异常值不敏感、无数据输入假定
* 缺点：计算复杂度高、空间复杂度高
* 适用数据范围：数值型和标称型

### 本节其他参考资料

* [K近邻法(KNN)原理小结 - 刘建平Pinard - 博客园](http://www.cnblogs.com/pinard/p/6061661.html)