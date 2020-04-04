[toc]

# CRF -- 条件随机场

## 阅读提示

**阅读提示**：先看[条件随机场（CRF）和隐马尔科夫模型（HMM）最大区别在哪里？CRF的全局最优体现在哪里？](https://www.zhihu.com/question/53458773)，然后[CRF系列](https://www.cnblogs.com/baiboy/p/crf2.html)，这些看完以后，这个[NLP —— 图模型（二）条件随机场（Conditional random field，CRF）](https://www.cnblogs.com/Determined22/p/6915730.html)作为补充，同时结合自己总结的HMM看，不过感觉还是云里雾里。



## CRF为什么用到NER上？

因为在NER好设计特征函数（可以简单地理解为规则，比如定义识别人名的规则）。如下为一个特征函数

![img](..\img\CRF2.png)

案例：讲了CRF分词，见课件。



## Reference

[1] 七月在线课件《条件随机场（初学版）》