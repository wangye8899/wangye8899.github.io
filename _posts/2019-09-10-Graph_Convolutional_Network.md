---
layout: post
title: Graph Convolutional Network
category: 技术
tags: [Graph Convolutional network]
description: 在阅读完有关于图神经网络、图卷积神经网络相关的论文之后，这篇post主要是对图卷积神经网络进行总结。
---


> 图卷积神经网络

对于Graph Convolution Network,我也是第一次接触，再看了几篇论文之后(准确的说应该是一片Survey和维基百科的内容)，终于对图卷积神经网络有了一些认识，并且也逐渐熟悉了其中蕴含的数学道理。不过古语有云“纸上得来终觉浅，绝知此事要躬行”，还是把自己的所得记录下来，形成文字。

## CNN和GraphCNN

众所周知，CNN作为基本的神经网络，其应用范围很广泛，包括图片、语音、语句，当然这些数据也对应着各自的任务，比如图片分类，语音识别，自然语言处理等。我们把包括图片、语音等这样的数据称为 *“the data of Euclidean domain”*，中文为： *欧几里得域的数据*。那么什么欧式数据呢？其实简单的对其进行概括和总结：就是是数据表示具有网格结构的数据,the data representatiion have a grid structure。那么CNN便可以通过卷积核的卷积运算充分的提取数据的特征，用作下一步模型的操作。但是，如果数据不具备这样的网格结构，CNN还适用吗？答案是否。

那么不具备网格结构的数据是什么数据呢？没错，这样的数据被称为 *“the data of Non-Euclidean domain”* ,非欧式域的数据。下面我列举了几种非欧式数据的例子：


![社交网络](https://ws1.sinaimg.cn/large/006CCxP6gy1g6urgp7jlfj30ae03raa2.jpg)

图1 社交网络

![3.jpeg](https://ws1.sinaimg.cn/large/006CCxP6ly1g6uriydy35j308504smx6.jpg)

图2 生物基因

![2.jpeg](https://ws1.sinaimg.cn/large/006CCxP6ly1g6urjh6i8vj308k04kdfs.jpg)

图3 论文引用网络



从图上数便可以看出，每一个数据的结构都是基于图结构的(包括节点和边)，那么类似于这样的数据，直观上便可以看出，不具备网格结构。那么这样的数据是无法直接应用CNN的，原因有很多，最主要的原因是，对于图上的每个节点，所使用的卷积核都不会是相同的。那么，这样的数据应该使用什么模型呢？GraphCNN。

*GraphCNN*全称为 *Graph Convolutional Netural Network*，*GraphCNN* 简要的理解就是把卷积算子应用到Graph上面，目前*GraphCNN*主要分为两种——Spectral domain(谱域)、Spatial domain(空间域)。这篇文章仅仅总结基于Spectral domain 的GraphCNN。


## 基于Spectral domain的GraphCNN
基于Spectral domain的思想，源于graph signal process理论。它的背后由大量严谨的数学原理作为支撑。同时，Spectral domain 理论中一个最大的假设：Graph is undirected。那么如何将卷积应用至Graph上呢，其中的蕴含的数学原理有哪些呢？以下内容，将逐步进行解释。

## 从CNN到GraphCNN的数学原理

>The definition of Convolution

当提及到这样的问题，可能最先想到的就是CNN中的卷积运算，没错，但CNN中的卷积也只是一种形式。对于卷积的数学定义：

设$f(x)$、$g(x)$是$R$上的两个可积函数，那么卷积定义为：

$$
f*g = \int_{-\infty}^{+\infty}f(x)g(y-x)dx
$$
那么从这个公式便可以知道，以前总讨论的卷积(尤指图片上的卷积操作)，只不过是在离散点数据上做乘积然后再做加法。如下图：


![](https://pic1.zhimg.com/50/v2-15fea61b768f7561648dbea164fcb75f_hd.webp)

图4 二维卷积操作


而把定义域扩大至实数，那么对于这种累加操作也就可以使用积分定义。所以上图的卷积只不过是公式定义ed其中一种子形式。

> How to apply convolution from Euclidean domain to Non-Euclidean domain

*Fourier transform*(傅里叶变换)

Fourier transform(傅里叶变换)是一种线性积分，用于信号处理中时域和频域之间的变换。在维基百科中，傅里叶变换主要的思想是：复杂的周期函数可以用一系列简单的正弦、余弦波之和表示。

数学定义：
$$
\hat{f}(\omega)=\int_{\mathbf{R}^{n}} f(x) e^{-i \omega \cdot x} d x
$$
其中 $\hat{f}(\omega)$ 是傅里叶变换后的结果，成为频谱。

傅里叶变换特性中包含卷积特性，这个性质对于在Graph domain中应用Convolution非常重要。维基百科中对于卷积特性是这样定义的：


![深度截图_选择区域_20190911100435.png](https://ws1.sinaimg.cn/large/006CCxP6gy1g6vcnl2ho1j319a068407.jpg)
图5 傅里叶变换的卷积特性




简言之，就是两个函数的卷积的傅里叶变换频谱相当于傅里叶域中这两个函数经过傅里叶变换后
的频谱的乘积。数学描述为：
$$
F(f*g) = F(f)F(g)
$$
那么如果我们对上述数学表达做一下简单的变换，即等式左右同时取逆。
$$
F^{-1}F(f*g) = F^{-1}(F(f)F(g))
$$
进而得到：
$$
f*g =  F^{-1}(F(f)F(g))
$$
那么综上所述，得出结论：
两个函数的卷积等于这两个函数在相应的傅里叶域的傅里叶变换频谱的乘积后取逆。

> How to get the Fourier transform of f and g in the graph domain

图傅里叶变换在维基百科上是这样定义的。


![深度截图_选择区域_20190911103813.png](https://ws1.sinaimg.cn/large/006CCxP6gy1g6vdmb0tbdj31550j3tbf.jpg)

图6 维基百科定义的图傅里叶变换


这里面，接触到了图的拉普拉斯矩阵。可以看到，上图中拉普拉斯矩阵的定义为：
$$
L = D - W 
$$
其中D表示图的度矩阵，W表示图的邻接矩阵。下图为例：


![深度截图_选择区域_20190911104453.png](https://ws1.sinaimg.cn/large/006CCxP6ly1g6vdt9hux9j31450bkmz8.jpg)
图7 $L = D - W$ 无向图拉普拉斯矩阵示例


那么我们可以发现，这样定义的拉普拉斯矩阵为对称矩阵。但在论文中，拉普拉斯矩阵的定义并不仅限于此，论文中使用的图拉普拉斯矩阵的定义如下：

$$
L^{\mathrm{sym}} :=D^{-\frac{1}{2}} L D^{-\frac{1}{2}}=I-D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
$$
对于此定义的推导过程如下：
$$
L^{sym} = D^{-1/2}LD^{-1/2}=D^{-1/2}(D-A)D^{-1/2}=I-D^{-1/2}AD^{-1/2}  
$$
$I$是单位矩阵。
不必对此感到奇怪，因为在维基百科上，对于拉普拉斯矩阵的定义本来就包含这种形式。这种形式体现了对称性。

回到图6，其中在得出拉普拉斯矩阵之后，又进行了特征分解(线性代数内容)，那么为什么拉普拉斯矩阵可以进行特征分解呢？有线性代数或者矩阵论的知识可知，拉普拉斯矩阵属于实对称矩阵，而实对称矩阵必然可以进行特征分解。则会得到以下式子：
$$
L=U\left(\begin{array}{ccc}{\lambda_{1}} & {} & {} \\ {} & {\ddots} & {} \\ {} & {} & {\lambda_{n}}\end{array}\right) U^{T}
$$
好，我们从线性代数的角度进行分析：

$U$是L矩阵经过特征分解后得到的特征向量，
$$
\left(\begin{array}{cccc}{\lambda_{1}} \\ {} & {} & {\ddots} \\ {} & {} & {} & {\lambda_{n}}\end{array}\right)
$$
是特征向量对应的特征值。

那么根据图6可知，$U^T$ 便等同于 $U^{-1}$ (线性代数正交矩阵基本知识)。此时，$U^T$ 便定义为此Graph上的图论傅里叶转换矩阵。那么对于向量:
$$
s=\left(s_{1}, s_{2}, \ldots s_{N}\right)^{T}
$$

$s_k$ 表示graph定点k的信号值。它的图论傅里叶变换为：

$$
\hat{s}=U^{-1}s=U^{T}s
$$
逆傅里叶变换为:
$$
s = U\hat{s}
$$

好的，写到这，我们就可以得到函数f和g在特定图上的图傅里叶变换：

$$
\hat{f}=U^{-1}f=U^{T}f
$$

$$
\hat{g}=U^{-1}g=U^{T}g
$$

>apply convolution in graph

由傅里叶变换的卷积特性可知：
$$
f*g =  F^{-1}(F(f)F(g))
$$
那么在已知两个函数的傅里叶变换后，如何求解卷积呢？很简单，如上公式所示:

*_只需要对频谱的乘积取逆即可_*
$$
f*g =  F^{-1}(F(f)F(g))=U(U^{-1}fU^{-1}g)=
\mathbf{U}\left(\mathbf{U}^{T} \mathbf{f} \odot \mathbf{U}^{T} \mathbf{g}\right)
$$

$\odot$ 表示[*Hadamard*积](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))。

## 总结
本文介绍了如何将Convolution应用至Graph上，主要的思想就是计算相应图的拉普拉斯矩阵，并对其进行特征分解，得到的特征向量便是图傅里叶变换的转换矩阵或者成为基。在转换矩阵(基)的帮助下，对输入的信号f和卷积函数g进行傅里叶变换分别得到对应的频谱。对频谱的乘积取逆运算后得到f和g在图上的卷积结果。



