## Logistic 回归

### 原理

利用现有数据对分类边界线建立回归公式，以此进行分类。

### 数学知识

Sigmoid 函数公式：

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Sigmoid 函数的输入记为 $z$，采用向量的写法为：

$$z = w^Tx$$

### 梯度上升的最优化方法

#### 梯度上升法

基于思想：要找到某函数的最大值，最好的方法是沿着该函数的梯度方向探寻。

梯度上升算法的迭代公式如下：

$$w:=w+\alpha \nabla\_wf(w)$$

其中，梯度算子 $\nabla\_wf(w)$ 总是指向函数值增长最快的方向；**步长** $\alpha$ 代表移动量的大小。

该公式将一直被迭代执行，直至达到某个停止条件为止，比如迭代次数到达某个指定值或算法达到某个可以允许的误差范围。

代码实现：

```py
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# Logistic 回归梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
```

#### 随机梯度上升法

梯度上升算法每次更新回归系数时都需要遍历整个数据集，当样本量和特征数较大时计算复杂度较高。一种改进方法是**一次仅用一个样本点来更新**，即**随机梯度上升法**。由于可以在新样本到来时对分类器进行增量式更新，因而随机梯度上升算法是一个**在线**学习算法。

代码实现：

```py
# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01    # 可以通过随迭代次数不断减小 alpha 来缓解波动
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights
```

### 总结

Logistic 回归：

* 优点：计算代价不高，易于理解和实现
* 缺点：容易欠拟合，分类精度可能不高
* 适用数据类型：数值型和标称型

对于缺失值，可以用 0 填充，在更新时不会影响系数的值。

<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
   tex2jax: {inlineMath: [ ['$', '$'] ],
         displayMath: [ ['$$', '$$']]}
 });
</script>

<script src="https://cdn.bootcss.com/mathjax/2.7.4/latest.js?config=default"></script>