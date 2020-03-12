# Attention机制 --基础篇

## Hard vs Soft[1]

注意力机制可以分为Hard Attention和Soft Attention。这两个的区别为Hard Attention关注的是`很小一部分区域`，而Soft Attention关注的是`更大的区域`。如机器翻译的一个例子：

我是小明 -> I am xiaoming

当翻译'I'时，Hard Attention只关注“我”，Soft Attention关注整个句子中的每个单词“我”“是”“小明”；Hard Attention计算量小一些，Soft Attention计算量大一些；`Hard Attention在翻译过程中采用的是one-hot方式来对位置信息进行编码，如在第1时刻，位置信息就是[1,0,0],第2时刻，位置信息就是[0,1,0]。`

## Glove vs Local[2]

`在Soft Attention中，又划分成两大阵营：Glove Attention和Local Attention。`同样以机器翻译为例，在翻译过程中：

Glove Attention关注整个句子；获取的信息更全面，但计算量大，因此效率低一些。

Local Attention关注周围的单词，即窗口（窗口的大小需要自行调整，`如果设置太小，可能会导致效果变得很差,论文中采用高斯分布来实现，如下`）
$$
\hat{a}_{i,j} = a_{i,j} \, e^{- \frac{(s - p_t)^2}{2 \sigma^2}}, \sigma = \frac{D}{2}
$$
同时Local Attention计算量小。

此外，`在Local Attention中，多了一个预测中心词`$p_t$`的过程，这可能忽略一些重要的词，但同时如果选择得当，那么会降低无关词的干扰，当然，所带来的收益并不大。`

`而考虑到NLP中问题的复杂性，如句子长短不一，相互之间可能存在依赖，因此后来的很多的论文[3][4]中大部分选择使用Glove Attention。`

* Glove Attention

![](http://ww1.sinaimg.cn/large/006gOeiSly1g0tereh268j30zk0k00te.jpg)

* Local Attention

![](http://ww1.sinaimg.cn/large/006gOeiSly1g0terwkhqmj30zk0k0wf1.jpg)

## Attention的本质思想[5]

![](http://ww1.sinaimg.cn/large/006gOeiSly1g0tf397umyj30kn08uq34.jpg)

在上图中，Query表示问题信息，在阅读理解中，其对应的是question，在机器翻译中，其对应的是上一时刻的输出向量$S_{t-1}$。

Key和Value在大部分情况下是相同的，一般指的是第$j(j=1...n,n表示当前句子总的单词个数)$个词的表示，在机器翻译中，key与value用第j个词的隐藏层输出$h_j$表示（即$key_j=value_j=h_j$）。

如上述Glove Attention图例，我们通过计算Query与各个key的相似性或者相关性来获得$a_{i,j}(i表示Query编号)$：
$$
\alpha_{i,j}=\frac{e^{score(Query_i,Key_j)}}{\sum_{k=1}^ne^{score(Query_i,Key_k)}}
$$
然后对$a_{i,j}$进行加权求和：
$$
c_i=\sum_{j=1}^n\alpha_{i,j}Value_j
$$
由以上公式可以看出，Attention机制的计算过程分为以下三个步骤：

1.**score函数**：计算query与key的相似性或者相关性，即$score(Query_i,Key_j)$.

2.**计算注意力权重**：通过softmax函数对值进行归一化处理获得注意力权重值，即$\alpha_{i,j}$的计算。

3.**加权求和得到注意力值**：通过注意力权重值对value进行加权求和，即$c_i$的计算。

`总的来说，Attention无论怎么变化，总是万变不离其宗。对于大多数Attention文章来说，其变化为Value、Key、Query的定义，以及sorce函数的计算方法。`下面我们来详细讨论一下。

## Score函数的选择[6]

score函数是为了计算Query和Key的相关性或者相似性。主要分为以下几种：

1.向量内积，`学习快，适合向量在同一空间，如Transformer。`
$$
score(Query, Key(j)) = Query \cdot Key(j)
$$
2.余弦相似度
$$
score(Query, Key(j)) = \frac{Query \cdot Key(j)}{||Query|| \cdot ||Key(j)||}
$$
3.`MLP(Mlutilayer Perceptron)网络`，灵活
$$
score(Query, Key(j)) = MLP(Query,  Key(j)) \\
general: score(Query, Key(j)) = Query \, W \, Key(j) \\
concat: score(Query, key(j)) = W \, [Query;Key(j) ]
$$

## Query,Key,Value的定义

对于一个 Attention 机制而言，定义好 Query， Key， Value 是至关重要的，这一点我个人认为是一个经验工程，看的多了，自然就懂了。 我这里简单举阅读理解与机器翻译的例子：

- 对于机器翻译而言，常见的是： Query 就是上一时刻 Decoder 的输出 $S_{i-1}$， 而Key，Value 是一样的，指的是 Encoder 中每个单词的上下文表示。
- 对于英语高考阅读理解而言， Query 可以是**问题的表示**，也可以是**问题+选项的表示**， 而对于Key， Value而言，往往也都是一样的，都指的是**文章**。而此时Attention的目的就是找出**文章**中与**问题**或**问题+选项**的相关片段，以此来判断我们的问题是否为正确的选项。

由此我们可以看出， Attention 中 Query， Key， Value 的定义都是很灵活的，不同的定义可能产生不一样的化学效果，比如 **Self-Attention** ，下面我就好好探讨探讨这一牛逼思想。

## Self-Attention[5]

在self-attention中，query,key,value都是相同的向量，都是单词的上下文表示，从而使序列自己关注自己，即
$$
Attention\, value=Attention(W_QX,W_KX,W_VX)
$$
其中，$Query=W_QX,Key=W_KX,Value=W_VX$

`Self-Attention可以说的最火的Attention模型了，在BERT中起了重要的作用。`

`它的内部含义是对序列本身做Attention,来获得序列内部的联系，如下图所示[7]`

![](http://ww1.sinaimg.cn/large/006gOeiSly1g0u0j6zj7hg30go0er1kx.gif)

`这其实类似于我们在Embedding层的时候采用LSTM来获取输入序列的上下文表示，但与LSTM不同的是，Self-Attention更能够把握句子中词与词的句法特征或语义特征，但是对于序列的位置信息不能很好地表示，这也是为什么Self-Attention需要一个Postition Embedding来对位置信息做一个补充，但对于一些位置信息比较敏感的任务，Position Embedding所带来的位置信息可能会不够。`

`之所以说这篇文章具有开创意义，是因为其将Attention用到了一个基础单元上，为取代LSTM提供了一种可能。`

## 参考文献

[1]Show,Attend and Tell:Neural Image Caption Generation with Visual Attention

[2]Effective Approaches to Attention-based Neural Machine Translation

[3]Neural Machine Translate by jointly Learning to Align and Translate

[4]Neural Responding Machine for Short-Text Conversation

[5]Attention is All You Need

6.[深度学习中的注意力机制](https://blog.csdn.net/qq_40027052/article/details/78421155)

7.[Attention机制详解（二）——Self-Attention与Transformer](https://zhuanlan.zhihu.com/p/47282410)

