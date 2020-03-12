# 预训练语言模型 - XLNet

奈何本人没文化，一句卧槽走天下

## 前言

又屠榜了，一开始，我以为只是噱头，直到鄙人打开了RACE排行榜，XLNet超过了BERT_LARGE。

我最近几个月来一直希望能够通过设计一个适合RACE数据集的Attention机制融合长距离文本实现终极反杀，然后，看到这一幕，我对Attention的必要性产生了严重的怀疑。

**虽然说我们依旧可以在XLNet上加上Attention来实现超越XLNet本身的效果，**`(XLNet竟然没有使用Attention???)`但这真的有意义吗，或者再过几个月，屠榜的消息又来了，那么设计精巧的Attention机制又被**大力出奇迹**的模型按在地上摩擦的时候，意义何在？

## XLNet为何如此之叼？

XLNet在多个任务上超过了BERT，我们先来看看XLNet是怎么做的，然后分析相对于BERT而言，XLNet为何会如此优秀。

推荐张俊林大佬的讲解，真的是深入浅出： [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

### 1. AR LM

见《预训练语言模型的发展》

### 2. AE LM

见《预训练语言模型的发展》

### 3. 如何AR + AE

**XLNer的出发点就是结合AR和AE，从而实现二者的互补，从结果来看，它做的很棒，我们先抠模型的细节，然后分析为什么这么做能否避免某种问题。**

### 4. 预训练模型： PLM



## Refrence

[1] XLNet: Generalized Autoregressive Pretraining for Language Understanding

[2] Transformer-XL: Attention Language Model Beyond a Fixed-Length Context

[3]  [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

