---
layout: post
title: Graph Attention Networks
category: 技术
tags: [GAT]
description: 
---
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

> Graph Attention Networks 图注意力网络

## 简介：

在上两篇的博客中，我主要讲解了GCN中卷积算子的推到过程并分享了第一次组会的讲稿，也会像关于GCN的内容。今天，继续分享一片对之前的论文进行改进的图神经网络模型——GAT(Graph Attention Networks)。在之前的文章中，GCN经历了第一代简单粗暴的对角化卷积核到第二代多项式以及切比雪夫多项式对卷积核进行数学逼近的改进，这一步的改进，是卷积核学习的参数大幅度下降，同时也不再需要对图拉普拉斯矩阵的特征分解，复杂度大幅度下降。到了第三代Semi，使用一阶切比雪夫多项式、采用新的优化算子，最终再一次改进了GCN。在这篇论文中[《Graph Attention Networks》](https://arxiv.org/abs/1710.10903)，利用Attention机制，又一次改善了基于卷积、数学逼近的方式。对于这种方式的简单理解：Attention机制可以学习当前节点的邻居节点的特征信息并且不需要提前知道图的结构(也就是不需要计算图的拉普拉斯矩阵)，该模型在处理inductive problems and transductive problems也有着很好的优势，本文也给出了一些实验证明。

## Why GAT ？

我们已经知道，CNN不能够直接应用于Graph数据上，这是由于数据的格式问题和CNN
的工作性质。那么同样的角度来看，为什么GAT会出现呢？它主要解决的问题是什么呢？我们知道，之前我们介绍的三代图神经网络，最基本的先决条件是：使用无向图的拉普拉斯矩阵。那么也就是说，对于这样的GCN，我们首先要知道其处理的图的结构也就是拉普拉斯矩阵。那么如果图的结构是不可见的呢？GAT应运而生。

## How GAT work ?
GAT全称为：Graph Attention Networks，其原理使用了Attention机制的思想来处理Grpah数据。简单总结为一句话：*GAT通过计算当前节点的邻居节点的特征，从而计算当前节点的隐藏状态值。* 下面将介绍GAT的实现细节。
GAT层的输入是一系列的节点特征，具体的格式如下：
$$
\mathbf{h}=\left\{\vec{h}_{1}, \vec{h}_{2}, \ldots, \vec{h}_{N}\right\}, \vec{h}_{i} \in \mathbb{R}^{F}
$$
其中，$N$ 是节点数。 $F$ 是每个节点的特征数。那么Attention层经过一些列的处理，将产生一系列新的节点特征，如下：
$$
\mathbf{h}^{\prime}=\left\{\vec{h}_{1}^{\prime}, \vec{h}_{2}^{\prime}, \ldots, \vec{h}_{N}^{\prime}\right\}, \vec{h}_{i}^{\prime}
$$
以上，从输入输出的角度，对Attention层有了宏观的理解。那么具体的细节将通过以下几个方面展示：

1. 权重参数化

为了能更好的将输入特征转换成更高级别的特征，模型在第一步中加入了一个线性学习函数，其实也就是用权重函数对输入特征参数化处理，如下所示：
$$
e_{i j}=a\left(\mathbf{W} \vec{h}_{i}, \mathbf{W} \vec{h}_{j}\right)
$$
其中，$W$ 表示可学习权重矩阵， $\vec{h}_{i}$ 和 $\vec{h}_{j}$ 是节点i、j的特征输入 ， $a$ 是图上所有节点共享的Attention机制，我觉得可以把它理解为一个函数。

$e_{i j}$ 是经过Attention机制运算后得到的Attention系数。 $e_{i j}$ 同时也表示了节点j的特征对于节点i的重要性。注意力机制允许每一个节点去关注其他节点的特征信息。而忽略掉所有的结构信息，但是并不是注意力机制不考虑图的结构信息，而是说图的结构信息在计算的过程中就应经被注入其中。

2. softmax规范化

$$a_{i,j}=softmax(e_{ij})=\frac{exp(e_{ij})}{\Sigma_{k\in N_i}exp(e_{ik})}$$
这个公式可以直接理解为进行了一步规范化操作，论文中提到这样做的目的是可以很容易的比较不同节点的注意力系数。
最终的形式：

![深度截图_选择区域_20190923162052.png](https://ws1.sinaimg.cn/large/006CCxP6ly1g79j0c5vclj30kh03smxh.jpg)

以上便是，注意力机制的注意力系数的计算过程。那么在得到注意力系数之后，将计算当前节点的输出特征值。下面结合论文中的示意图，进行详细的阐述。

![深度截图_选择区域_20190923182009.png](https://ws1.sinaimg.cn/large/006CCxP6gy1g79meo59dsj318g0h30wi.jpg)
上图便是GAT的工作示意图，我们在这里将分左右顺序进行讲解:
对于左部分图，其数学表达式为：
$$
\vec{h}_{i}^{\prime}=\sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j} \mathbf{W} \vec{h}_{j}\right)
$$
其中 $\sigma$ 表示非线性函数，W是可学习权重，$\alpha_{i j}$ 是节点i、j之间的注意力系数。$\vec{h}_{j}$ 是j节点的特征信息。那么将所有处于节点i邻域内的点全部进行加和操作，最后使用非线性函数进行映射。将得到当前节点的隐藏特征值。

对于图右半部分，其数学表达式为：
$$
\vec{h}_{i}^{\prime}=\|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \vec{h}_{j}\right)
$$

右面的图例实际是对左面图例的改进，论文中，经过大量实验发现，当使用multi-head（多头）注意力机制时，取得效果要好于前者。理解起来也很容易，就是K个不同Attention同时进行相关的数学运算，最终将其各自的结果连接在一起。

GAT的原理和数学推导都比较简单，所以理解起来并不难。接下来，将介绍该论文中的实验部分。
___

## Experiment
本文中，作者强调了GAT模型能够解决transductive tasks and inductive tasks，也就是推理任务和归纳任务。同时，文章中也分别在不同的数据集上进行了实验。数据集情况如下：
![深度截图_选择区域_20190923184241.png](https://ws1.sinaimg.cn/large/006CCxP6ly1g79n24ari2j30xb0ctjti.jpg)

transductive tasks 实验结果

![深度截图_选择区域_20190923191139.png](https://ws1.sinaimg.cn/large/006CCxP6ly1g79nw9j16aj30t40eydja.jpg)

inductive tasks 实验结果
![深度截图_选择区域_20190923191256.png](https://ws1.sinaimg.cn/large/006CCxP6ly1g79nxlf3mjj30n20cvtae.jpg)

## The advantages of GAT

对比之前的实验，GAT主要有以下优势：
1. 运算高效

对于单个图上节点来讲，其与邻域中所有点的注意力系数的计算均是同步进行的。

对于所有的节点来说，它们的输出特征的计算也都是平行的。

在GAT的运算过程中，并不存在对于Graph拉普拉斯矩阵的特征分解。

2. 权重分配

在GAT中，对于图上的节点所分配的权重也都不一样，换句话说，可以根据实际数据的需要，对图上不同的节点赋以不同的权重。这点跟以前的网络相比，将会是极大的提升。

3. 共享注意力机制

GAT中的注意力机制以一种共享的方式应用于图上所有的边。故不需要提前了解Graph的结构。

