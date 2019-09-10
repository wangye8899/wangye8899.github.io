---
layout: post
title: Graph Convolutional Network
category: 技术
tags: [Graph Convolutional network]
description: 在阅读完有关于图神经网络、图卷积神经网络相关的论文之后，这篇post主要是对图卷积神经网络进行总结。
---


> 图卷积神经网络

对于Graph Convolution Network,我也是第一次接触，再看了几篇论文之后(准确的说应该是一片Survey和维基百科的内容)，终于对图卷积神经网络有了一些认识，并且也逐渐熟悉了其中蕴含的数学道理。不过古语有云“纸上得来终觉浅，绝知此事要躬行”，还是把自己的所得记录下来，形成文字。

## CNN和GraphNN

众所周知，CNN作为基本的神经网络，其应用范围很广泛，包括图片、语音、语句，当然这些数据也对应着各自的任务，比如图片分类，语音识别，自然语言处理等。我们把包括图片、语音等这样的数据称为 *“the data of Euclidean domain”*，中文为： *欧几里得域的数据*。那么什么欧式数据呢？其实简单的对其进行概括和总结：就是是数据表示具有网格结构的数据,the data representatiion have a grid structure。那么CNN便可以通过卷积核的卷积运算充分的提取数据的特征，用作下一步模型的操作。但是，如果数据不具备这样的网格结构，CNN还适用吗？答案是否。

那么不具备网格结构的数据是什么数据呢？没错，这样的数据被称为 *“the data of Non-Euclidean domain”* ,非欧式域的数据。下面我列举了几种非欧式数据的例子：

1. 社交网络

![社交网络](https://ws1.sinaimg.cn/large/006CCxP6gy1g6urgp7jlfj30ae03raa2.jpg)

2. 生物基因

![3.jpeg](https://ws1.sinaimg.cn/large/006CCxP6ly1g6uriydy35j308504smx6.jpg)

3. 论文引用网络

![2.jpeg](https://ws1.sinaimg.cn/large/006CCxP6ly1g6urjh6i8vj308k04kdfs.jpg)


从图上数便可以看出，每一个数据的结构都是基于图结构的(包括节点和边)，那么类似于这样的数据，直观上便可以看出，不具备网格结构。那么这样的数据是无法直接应用CNN的，原因有很多，最主要的原因是，对于图上的每个节点，所使用的卷积核都不会是相同的。那么，这样的数据应该使用什么模型呢？GraphNN。
