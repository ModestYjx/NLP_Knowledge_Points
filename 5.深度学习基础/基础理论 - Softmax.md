# 基础理论 - Softmax

---

## 1. Softmax 定义

$$
P(i) = \frac{e^{a_i}}{\sum_{k=1}^T e^{a_k}} \in [0,1]
$$





## 2. Softmax 损失

$$
L = - \sum_{j=1}^T y_j \, log \, s_j  \\
$$

---

## QA

### 1. 为何一般选择 softmax 为多分类的输出层

虽然能够将输出范围概率限制在 [0,1]之间的方法有很多，但 Softmax 的好处在于， 它使得**输出两极化**：正样本的结果趋近于 1， 负样本的结果趋近于 0。 可以说， Softmax 是 logistic 的一种泛化。

![](https://gss0.baidu.com/7Po3dSag_xI4khGko9WTAnF6hhy/zhidao/wh%3D600%2C800/sign=857c395dcb95d143da23ec2543c0ae3a/9d82d158ccbf6c8147d1f9b6bf3eb13532fa408e.jpg)

## 2.softmax溢出

[softmax](https://mp.weixin.qq.com/s?__biz=MzUxMjE1NzA2MA==&mid=2247484250&idx=1&sn=d7a66376604db810a4a0d0b5abf174d2&chksm=f969f7b1ce1e7ea7a618f502ee080c95b88cf6ddc41825a7f1f6c0c8ec795cca3ab482509a1b&mpshare=1&scene=23&srcid=&sharer_sharetime=1584708583379&sharer_shareid=f2db016cbffd15b40e734dd9b7a4efeb#rd)