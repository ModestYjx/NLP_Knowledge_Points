[toc]

# Boosting与Bagging

## 阅读提示

先看[Bagging，Boosting二者之间的区别](https://zhuanlan.zhihu.com/p/36848643)，再看[为什么说bagging是减少variance，而boosting是减少bias?](https://www.zhihu.com/question/26760839/answer/40337791)，然后看[GBDT算法原理深入解析](https://www.zybuluo.com/yxd/note/611571)。



## 集成学习中的包含关系，见[GBDT算法原理深入解析](https://www.zybuluo.com/yxd/note/611571)

集成学习（ensemble learning）

* Boosting
  * 梯度提升（Gradient boosting）
  * AdaBoost
* Bagging
* Stacking



下面是将决策树与这些算法框架进行结合所得到的新的算法：

1. Bagging + 决策树 = 随机森林
2. AdaBoost + 决策树 = 提升树
3. Gradient Boosting + 决策树 = GBDT