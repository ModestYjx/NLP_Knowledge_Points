# SVM

## QA

### 0. SVM了解吗？有什么优点？优化方法？



### 1. SVM 中的支持向量是什么意思？

![10.svm](C:/Users/Administrator/Desktop/笔记/NLPer-Interview-master/img/10.svm.png)

 如上图所示，我们在获得分离超平面时，并非所有的点都对分离超平面的位置起决定作用。

其实在特别远的区域，哪怕你增加10000个样本点，对于超平面的位置，也是没有作用的，因为分割线是由几个关键点决定的（图上三个），这几个关键点支撑起了一个分离超平面，所以这些关键点，就是**支持向量**。

### 2.  什么是SVM ？

SVM 是一种二类分类模型。它的基本思想是在特征空间中寻找间隔最大的分离超平面使数据得到高效的二分类， 主要分三种情况：

- 当训练样本线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机；
- 当训练数据近似线性可分时，引入松弛变量，通过软间隔最大化，学习一个线性分类器，即线性支持向量机
- 当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。

### 3. SVM 为何采用间隔最大化？

当训练数据线性可分时，存在无穷个分离超平面可以将两类数据正确分开。

- 感知机利用误分类最小策略，求得分离超平面，不过此时的解有无穷多个。
- 线性可分支持向量机利用间隔最大化求得最优分离超平面，这时，解是唯一的。

SVM 求得的分隔超平面所产生的分类结果是最鲁棒的，对未知实例的泛化能力最强。

### 4. 为何要讲求解 SVM 的原始问题转换为其对偶问题？

- 对偶问题往往更易求解

  当我们寻找约束存在时的最优点的时候，约束的存在虽然减小了需要搜寻的范围，但是却使问题变得更加复杂。为了使问题变得易于处理，我们的方法是把目标函数和约束全部融入一个新的函数，即拉格朗日函数，再通过这个函数来寻找最优点。

- 自然引入核函数，进而推广到非线性分类问题

### 5. SVM 与 LR 的区别

- LR是参数模型，SVM为非参数模型。
- LR采用的损失函数为logisticalloss（和交叉熵的形式一样），而SVM采用的是hingeloss([怎么样理解SVM中的hinge-loss：前两三个回答都可以](https://www.zhihu.com/question/47746939/answer/154058298))。
- 在学习分类器的时候，SVM只考虑与分类最相关的少数支持向量点。
- LR的模型相对简单，在进行大规模线性分类时比较方便。
- 从目标函数来看，区别在于逻辑回归采用的是logistical loss，SVM采用的是hinge loss。这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。
- SVM的处理方法是只考虑支持向量，也就是和分类最相关的少数点，去学习分类器。而逻辑回归通过**非线性映射**，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。
- 逻辑回归相对来说模型更简单，好理解，特别是大规模线性分类时比较方便。而SVM的理解和优化相对来说复杂一些，SVM转化为对偶问题后，分类只需要计算与少数几个支持向量的距离，这个在进行复杂核函数计算时优势很明显，能够大大简化模型和计算。
- logic 能做的 svm能做，但可能在准确率上有问题，svm能做的logic有的做不了。

### 6. SVM如何处理多分类问题？

- 直接法：直接在目标函数上修改，将多个分类面的参数求解合并到一个最优化问题里面。看似简单但是计算量却非常的大。
- 间接法：对训练器进行组合。其中比较典型的有一对一，和一对多。

一对多： 对每个类都训练出一个分类器，由svm是二分类，所以将此而分类器的两类设定为目标类为一类，其余类为另外一类。这样针对k个类可以训练出k个分类器，当有一个新的样本来的时候，用这k个分类器来测试，那个分类器的概率高，那么这个样本就属于哪一类。这种方法效果不太好，bias 比较高。

一对一： 针对任意两个类训练出一个分类器，如果有 k 类，一共训练出 $C_k^2$ 个分类器，这样当有一个新的样本要来的时候，用这 $C_k^2$ 个分类器来测试，每当被判定属于某一类的时候，该类就加一，最后票数最多的类别被认定为该样本的类。

### 5. SVM 软间隔与硬间隔表达式

- 硬间隔：
  $$
  min_{w,b} \frac{1}{2} ||w||^2 \qquad st. \quad y^{(i)}(w^Tx^{(i)} + b) \geq 1
  $$

- 软间隔：
  $$
  min_{w,b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^m \xi_i  \qquad st. \quad y^{(i)}(w^Tx^{(i)} + b) \geq 1 \quad \xi_i \geq 0
  $$


### 11. 核函数的种类和应用场景

- 线性核函数：主要用于线性可分的情形。参数少，速度快。
- 多项式核函数：
- 高斯核函数：主要用于线性不可分的情形。参数多，分类结果非常依赖于参数。
- sigmoid 核函数：
- 拉普拉斯核函数：

如果feature数量很大，跟样本数量差不多，建议使用LR或者 Linear kernel 的SVM。
如果 feature 数量较少，样本数量一般，建议使用 Gaussian Kernel 的SVM。

### 12. SVM 损失函数是什么([怎么样理解SVM中的hinge-loss：前两三个回答都可以](https://www.zhihu.com/question/47746939/answer/154058298))？

$$
J(\theta) = \frac{1}{2} ||\theta||^2 + C \sum_i max(0, 1-y_i(\theta^Tx_i + b))
$$

### 13. 核函数的作用是啥？

核函数能够将特征从低维空间映射到高维空间， 这个映射可以把低维空间中不可分的两类点变成线性可分的。

### 14. SVM 为何能用对偶函数求解？

对偶将原始问题中的约束转为了对偶问题中的等式约束， 而且更加方便了核函数的引入， 同时也改变了问题的复杂度， 在原始问题下， 求解问题的复杂度只与样本的维度有关， 在对偶问题下， 只与样本的数量有关。

### 15. 为什么高斯核能够拟合无穷维度？

因为将泰勒展开式代入高斯核，将会得到一个无穷维度的映射。

## 简介

SVM关键词：间隔、**对偶**、**核技巧**（`具体怎么使用还不清楚？`），它属于**判别模型**。

- **支持向量：**在求解的过程中，会发现只根据部分数据就可以确定分类器，这些数据称为支持向量。
- **支持向量机（SVM）**：其含义是通过**支持向量**运算的分类器。

SVM 是一种**二分类模型**， 它的目的是寻找一个超平面来对样本进行分割，分割的依据是**间隔最大化**，最终转化为一个**凸二次规划问题**来求解。

## 1. 线性可分

$D_0$ 和 $D_1$ 是 n 维空间中的两个点集， 如果存在 n 维向量 $w$ 和实数 $b$ ， 使得：
$$
wx_i +b > 0; \quad x_i \in D_0 \\
wx_j + b < 0; \quad x_j \in D_1
$$
则称 $D_0$ 与 $D_1$ 线性可分。

## 2. 最大间隔超平面

能够将 $D_0$  与 $D_1$ 完全正确分开的 $wx+b = 0$ 就成了一个超平面。

为了使得这个超平面更具鲁棒性，我们会去找最佳超平面，以最大间隔把两类样本分开的超平面，也称之为**最大间隔超平面**。

- 两类样本分别分割在该超平面的两侧
- 两侧距离超平面最近的样本点到超平面的距离被最大化了

## 3. 什么是支持向量？

**训练数据集中与分离超平面距离最近的样本点成为支持向量**

![1](C:/Users/Administrator/Desktop/笔记/NLPer-Interview-master/img/SVM/1.jpg)

## 4. SVM 能解决哪些问题？

- **线性分类：**对于n维数据，SVM 的目标是找到一个 n-1 维的最佳超平面来将数据分成两部分。 

  通过增加一个约束条件： **要求这个超平面到每边最近数据点的距离是最大的。**

- **非线性分类：** SVM通过结合使用**拉格朗日乘子法**和KTT条件（[浅谈最优化问题的KKT条件](https://zhuanlan.zhihu.com/p/26514613)），以及**核函数**可以生产非线性分类器

另一种回答方式：具体题目记不清了，大概意思是说SVM不仅能用于做分类任务，也可以做回归任务等。

### 5. 支持向量机的分类

- **硬间隔SVM（线性可分SVM)**： 当训练数据线性可分时，通过间隔最大化，学习一个线性表分类器。
- **软间隔SVM(线性SVM)**：当训练数据接近线性可分时，通过软间隔最大化，学习一个线性分类器。
- **Kernel SVM**： 当训练数据线性不可分时，通过使用**核技巧及软间隔最大化**，学习非线性SVM。

必会：硬间隔最大化 --> 学习的对偶问题 --> 软间隔最大化 --> 非线性支持向量机（核技巧）（`具体怎么使用？只可以核技巧可以用来将非线性问题映射为线性问题`）

### 硬间隔SVM

### 0. 几何间隔与函数间隔



### 1. SVM 最优化问题

任意分离超平面可定义为：
$$
w^Tx + b = 0
$$
二维空间中点 $(x,y)$ 到直线 $Ax + By + C=0$ 的距离公式为：
$$
\frac{|Ax+ By + C|}{\sqrt{A^2 + B^2}}
$$


扩展到n维空间中，任意点 $x$  到超平面$w^Tx + b = 0$ 的距离为：
$$
\frac{|w^Tx + b|}{||w||} \\
||w|| = \sqrt{w_1^2 + ... + w_n^2}
$$
假设，支持向量到超平面的距离为 $d$ ，那么就有：
$$
\begin{cases} \frac{w^Tx_i + b}{||w||} \geq d, & y_i = +1 \\ \frac{w^Tx_i + b}{||w||} \leq -d, & y_i = -1 \end{cases}
$$
稍作转化可得到：
$$
\begin{cases} \frac{w^Tx_i + b}{||w||d} \geq 1, & y_i = +1 \\ \frac{w^Tx_i + b}{||w||d} \leq -1, & y_i = -1 \end{cases}
$$
考虑到 $||w||d$ 为正数，我们暂且令它为 1（之所以令它等于 1，是为了方便推导和优化，且这样做对目标函数的优化没有影响，`因为我们只考虑目标函数的正负号？那目标函数是什么？不是间隔最小化吗？`），另一种推导方式见西瓜书（详细推导见“南瓜书”）：
$$
\begin{cases} w^Tx_i + b >= +1, & y_i = +1 \\ w^Tx_i + b <= -1, & y_i = -1 \end{cases}
$$

两个方程合并，则有**以下限定条件**：
$$
y_i (w^Tx_i + b) \geq 1
$$
那么我们就得到了最大间隔超平面的上下两个超平面：

![2](C:/Users/Administrator/Desktop/笔记/NLPer-Interview-master/img/SVM/2.jpg)

两个异类超平面的公式分别为：
$$
\begin{cases} w^Tx_i + b = +1, & y_i = +1 \\ w^Tx_i + b = -1, & y_i = -1 \end{cases}
$$
那么两个异类超平面之间的间隔为：
$$
\frac{2}{||w||}
$$
我们的目的是最大化这种间隔：
$$
\begin{align}
max \quad \frac{2}{||w||} &:= min \quad \frac{1}{2} ||w|| \\
&:= min \quad \frac{1}{2} ||w||^2
\end{align}
$$
那么我们的最优化问题为：
$$
min \quad \frac{1}{2} ||w||^2 \quad \\ st. y_i(w^Tx_i + b) \geq 1
$$

### 2. 对偶问题（[如何理解对偶问题](为什么我们要考虑线性规划的对偶问题？ - 运筹OR帷幄的回答 - 知乎 https://www.zhihu.com/question/26658861/answer/631753103)）

#### 1. 拉格朗日乘数法 - 等式约束优化问题

高等数学中，其等式约束优化问题为：
$$
min \, f(x_1, ..., x_n) \quad  st. \quad h_k(x_1, ... , x_n) = 0
$$
那么令：
$$
L(x, \lambda) = f(x) + \sum_{k=1}^l \lambda_k h_k(x)
$$

- $L(x, \lambda)$ ： Lagrange 函数
- $\lambda$ ： Lagrange 乘子，没有非负要求

利用必要条件找到可能的极值点，我们得到如下的方程组：
$$
\begin{cases} 
\frac{\delta L}{ \delta x_i} = 0, & i=1,2,...,n \\ 
\frac{\delta L}{ \delta \lambda_k} = 0, & k=1,2,...,l
\end{cases}
$$
等式约束下的Lagrange 乘数法引入了 $l$ 个 Lagrange 乘子，我们将 $x_i$ 与 $\lambda_k$ 一视同仁，将$\lambda_k$ 也看做优化变量，那么共有 $(n+l)$ 个优化变量。

#### 2. 拉格朗日乘数法 - 不等式约束优化问题

对于不等式约束优化问题，其主要思想在于**将不等式约束条件转变为等式约束条件，引入松弛变量，将松弛变量也是为优化变量。**

![3](C:/Users/Administrator/Desktop/笔记/NLPer-Interview-master/img/SVM/3.jpg)

对于我们的问题：
$$
min \quad \frac{1}{2} ||w||^2 \quad \\ st. \quad  g_i(w) = 1- y_i(w^Tx_i + b) \leq 0
$$
引入松弛变量 $a_i^2$ 得到：
$$
f(w) =  \frac{1}{2} ||w||^2 \\
g_i(w) = 1- y_i(w^Tx_i + b)  \\
h_i(w, a_i) = g_i(w) + a_i^2 = 0
$$
这里加平方主要为了不再引入新的约束条件，如果只引入 $a_i$ 那我们必须要保证 $a_i \geq 0$ 才能保证 $h_i(w, a_i)$ ，这不符合我们的意愿。

此时，我们就将不等式约束转化为等式约束，并得到 Lagrange 函数（见西瓜书的6.8）：
$$
\begin{align}
L(w, \lambda, a) &= \frac{1}{2}||w||^2  + \sum_{i=1}^n \lambda_i h_i(w) \\
&= \frac{1}{2}||w||^2 + \sum_{i=1}^n \lambda_i [g_i(w) + a_i^2] \quad \lambda_i \geq 0
\end{align}
$$
那么我们得到方程组有：
$$
\begin{cases} 
\frac{\delta L}{ \delta w_i} =\frac{\delta f}{\delta w_i} + \sum_{i=1}^n \lambda_i \frac{\delta g_i}{\delta w_i} =0 \\ 
\frac{\delta L}{ \delta a_i} = 2 \lambda_i a_i =0 \\
\frac{\delta L}{ \delta \lambda_i}=g_i(w) + a_i^2 = 0 \\
\lambda_i \geq 0  \qquad
\end{cases}
$$
针对 $\lambda_i a_i = 0$ 有两种情况：

-  $\lambda_i = 0, a_i \neq 0$：此时约束条件 $g_i(w)$ 不起作用且 $g_i(w) < 0$ 

-  $\lambda_i \neq 0, a_i = 0$： 此时 $g_i(w)=0, \lambda_i > 0$， 可以理解为约束条件 $g_i(w)$ 起作用了， 且$g_i(w) = 0$

综合可得：$\lambda_ig_i(w) = 0$， 且在约束条件起作用时 $\lambda_i > 0, g_i(w) = 0 $； 约束不起作用时， $\lambda_i = 0, g_i(w) < 0 $。

此时，方程组转化为：
$$
\begin{cases} 
\frac{\delta L}{ \delta w_i} =\frac{\delta f}{\delta w_i} + \sum_{i=j}^n \lambda_j \frac{\delta g_i}{\delta w_i} =0 \\ 
\lambda_ig_i(w) = 0 \\
g_i(w) \leq 0 \\
\lambda_i \geq 0
\end{cases}
$$
以上便是不等式约束优化优化问题的 **KKT(Karush-Kuhn-Tucker) 条件**， $\lambda_i$ 称为 KKT 乘子。

KTT 条件中，对于不同样本点来说

- 支持向量 $g_i(w) = 0$， 此时 $\lambda_i > 0$ 即可
- 其余向量 $g_i(w) < 0$， 此时 $\lambda_i = 0$

我们原问题是求$min\frac{1}{2}||w||^2$，现在即求$minL(w,\lambda,a)$（`这点没搞清楚，先假设就是这样吧？`）：
$$
\begin{align}
L(w, \lambda, a) &=  f(w) + \sum_{i=1}^n \lambda_i h_i(w) \\
&= f(w) + \sum_{i=1}^n \lambda_i [g_i(w) + a_i^2] \\
&= f(w) + \sum_{i=1}^n \lambda_i g_i(w) + \sum_{i=1}^n \lambda_i a_i^2
\end{align}
$$
由于 $\sum_{i=1}^n \lambda_ia_i^2 \geq 0$， 那么问题可以转化为：
$$
\begin{align}
L(w, \lambda) &=  f(w) + \sum_{i=1}^n \lambda_i g_i(w) \\
&=  \frac{1}{2} ||w||^2 + \sum_{i=1}^n \lambda_i (1- y_i(w^Tx_i + b) )
\end{align}
$$
假设我们找到了最佳的参数 $w$ 使得 $\frac{1}{2} ||w||^2 = p$ ，又因为 $\sum_{i=1}^n \lambda_i (1- y_i(w^Tx + b) ) \leq 0$， 因此有 $L(w, \lambda) \leq p$， 我们需要找到最佳的参数 $\lambda$， 使得 $L(w, \lambda)$ 接近 p， 此时问题转化为$max_{\lambda} \, L(w, \lambda)$，故我们的最优化问题转换为：
$$
min_wmax_{\lambda} \, L(w, \lambda) \quad \\ s.t. \quad \lambda_i \geq 0
$$

### 3. 引入对偶问题 -- TODO

- 弱对偶性： 最大的里面挑出来的最小的也要比最小的里面挑出来的最大的要大（一个不恰当的例子：清华的最弱也比中南的最强的厉害）
  $$
  min \, max \, f \geq max \, min \, f
  $$

- 强对偶性：KKT 条件是强对偶性的充要条件。

### 4. SVM 优化

- SVM 的优化问题为：
  $$
  min \quad \frac{1}{2} ||w||^2 \quad \\ st. \quad  g_i(w) = 1- y_i(w^Tx_i + b) \leq 0
  $$




- 构造拉格朗日函数：
  $$
  min_{w,b}max_{\lambda} L(w, b, \lambda) = \frac{1}{2} ||w||^2 + \sum_{i=1}^n \lambda_i (1- y_i(w^Tx_i + b) ) \\
  s.t. \lambda_i \geq 0
  $$

- 利用强对偶性转化：
  $$
  max_{\lambda}min_{w,b} \, L(w, b, \lambda)
  $$
  对参数 $w, b$ 求偏导有（`从这往后就看迷糊了？不知道为什么要这样做？只看到它要这样做`）：
  $$
  \frac{\delta L}{\delta w} = w - \sum_{i=1}^n \lambda_i x_i y_i = 0 \\
  \frac{\delta L}{\delta b} = \sum_{i=1}^n \lambda_i y_i = 0
  $$
  得到：
  $$
  w =  \sum_{i=1}^n \lambda_i x_i y_i \\
  \sum_{i=1}^n \lambda_i y_i = 0
  $$
  将两式带入到 $L(w, b, \lambda)$ 中有：
  $$
  \begin{align}
  min_{w,b} \, L(w, b, \lambda) &=\sum_{j=1}^n \lambda_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j y_i y_j x_i^Tx_j
  \end{align}
  $$

- 求解模型：

### 2. 软间隔SVM

![](https://pic3.zhimg.com/80/v2-834c7a5f310e187b448831676b7eeeee_720w.jpg)

软间隔允许部分样本点不满足约束条件：
$$
1 - y_i (w^Tx_i + b) \leq 0
$$

即相比于硬间隔的苛刻条件，我们允许个别样本点出现在间隔带里面，为了度量这个间隔软到何种程度，我们为每个样本引入一个**松弛变量**$\xi_i$，令$\xi_i>=0$，而且
$$
1 - y_i (w^Tx_i + b) -\xi_i\leq 0
$$
![](https://pic1.zhimg.com/80/v2-8e3d96fd9f9cad298628c7e2c4c8a8b8_720w.jpg)
增加软间隔后我们的优化目标变成了：

![](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7Bw%7D+%5Cfrac%7B1%7D%7B2%7D+%7C%7Cw%7C%7C%5E2+%2B+C%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Cxi_i+%5C%5C+s.t.%5Cquad+g_i%28w%2Cb%29+%3D+1+-+y_i%28w%5ETx_i%2Bb%29+-+%5Cxi_i%5Cleq+0%2C+%5Cquad+%5Cxi_i+%5Cgeq+0%2C+%5Cquad+i%3D1%2C2%2C...%2Cn+%5C%5C)

其中 C 是一个大于 0 的常数，可以理解为错误样本的惩罚程度，若 C 为无穷大， ![[公式]](https://www.zhihu.com/equation?tex=%5Cxi_%7Bi%7D) 必然无穷小，如此一来线性 SVM 就又变成了线性可分 SVM；当 C 为有限值的时候，才会允许部分样本不遵循约束条件。

## Kernel SVM

### 1. 思想

**对于在有限维度向量空间中线性不可分的样本，我们将其映射到更高维度的向量空间里，再通过间隔最大化的方式，学习得到支持向量机，就是非线性 SVM。**

用 x 表示原来的样本点，用 $\phi(x)$ 表示 x 映射到新特征空间后的新向量。那么分割超平面可以表示为：
$$
f(x) = w \phi (x) + b
$$
此时，非线性 SVM 的对偶问题转化为（`黑人问号？？？`）：

![](https://www.zhihu.com/equation?tex=%5Cmin%5Climits_%7B%5Clambda%7D+%5B%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Clambda_i+%5Clambda_j+y_i+y_j+%28%5Cphi%28x_i%29+%5Ccdot+%5Cphi%28x_j%29%29-%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Clambda_i%5D+%5C%5C+s.t.++%5Cquad+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Clambda_iy_i+%3D+0%2C+%5Cquad+%5Clambda_i+%5Cgeq+0%2C+%5Cquad+C-%5Clambda_i-%5Cmu_i%3D0+%5C%5C)

可以看到与线性 SVM 唯一的不同就是：之前的 ![[公式]](https://www.zhihu.com/equation?tex=%28x_i+%5Ccdot+x_j%29) 变成了 ![[公式]](https://www.zhihu.com/equation?tex=%28%5Cphi%28x_i%29+%5Ccdot+%5Cphi%28x_j%29%29) 。

### 2. 核函数的作用

- 目的： 将原坐标系中线性不可分数据通过核函数映射到另一空间，尽量使数据在新的空间里线性可分。**同时减小计算量**：

  这是因为低维空间映射到高维空间后维度可能会很大，如果将全部样本的点乘全部计算好，这样的计算量太大了。

  但如果我们有这样的一核函数 ![[公式]](https://www.zhihu.com/equation?tex=k%28x%2Cy%29+%3D+%28%5Cphi%28x%29%2C%5Cphi%28y%29%29) ， ![[公式]](https://www.zhihu.com/equation?tex=x_i) 与 ![[公式]](https://www.zhihu.com/equation?tex=x_j) 在特征空间的内积等于它们在原始样本空间中通过函数 ![[公式]](https://www.zhihu.com/equation?tex=k%28+x%2C+y%29) 计算的结果，我们就不需要计算高维甚至无穷维空间的内积了。

### 2. 常见核函数

我们常用核函数有：

**线性核函数**

![[公式]](https://www.zhihu.com/equation?tex=k%28x_i%2Cx_j%29+%3D+x_i%5ETx_j+%5C%5C)

**多项式核函数**

![[公式]](https://www.zhihu.com/equation?tex=+k%28x_i%2Cx_j%29+%3D+%28x_i%5ETx_j%29%5Ed%5C%5C)

**高斯核函数**

![[公式]](https://www.zhihu.com/equation?tex=k%28x_i%2Cx_j%29+%3D+exp%28-%5Cfrac%7B%7C%7Cx_i-x_j%7C%7C%7D%7B2%5Cdelta%5E2%7D%29+%5C%5C)

这三个常用的核函数中只有高斯核函数是需要调参的。

## SVM优缺点

优点：

- 有严格的数学理论支持，可解释性强，不依靠统计方法，从而简化了通常的分类和回归问题；
- 能找出对任务至关重要的关键样本（即：支持向量）；
- **采用核技巧之后，可以处理非线性分类/回归任务；**
- 最终决策函数只由少数的支持向量所确定，计算的复杂性取决于支持向量的数目，而不是样本空间的维数，这在某种意义上避免了“维数灾难”

缺点：

- 训练时间长。当采用 SMO 算法时，由于每次都需要挑选一对参数，因此时间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28N%5E2%29) ，其中 N 为训练样本的数量（`???`）；
- 当采用核技巧时，如果需要存储核矩阵，则空间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28N%5E2%29) (`???`)；
- 模型预测时，预测时间与支持向量的个数成正比。当支持向量的数量较大时，预测计算复杂度较高。

因此支持向量机目前只适合小批量样本的任务，**无法适应百万甚至上亿样本的任务**。

## Reference

[1] [支持向量机 SVM](https://zhuanlan.zhihu.com/p/77750026)