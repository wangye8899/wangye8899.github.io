---
layout: post
title: Graph Convolutional Network
category: 技术
tags: [Graph Convolutional network]
description: 在阅读完有关于图神经网络、图卷积神经网络相关的论文之后，这篇post主要是对图卷积神经网络进行总结。
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

>图神经网络
这篇博客，是根据我在组会上讲解GNN的讲稿编辑而成，内容较多，我直接转成了jpg，上传上来。文末附上下载链接。

![0_01.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y640lfvj30zk0k0myp.jpg)
![0_02.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y6tdsdxj30zk0k0q4e.jpg)
![0_03.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y7qgosrj30zk0k03zb.jpg)
![0_04.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y7vglfdj30zk0k0jw5.jpg)
![0_05.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y849ktrj30zk0k0n1e.jpg)
![0_06.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y888dduj30zk0k0779.jpg)
![0_07.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y8hrlsej30zk0k0tbm.jpg)
![0_08.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y8mfalxj30zk0k0gmf.jpg)
![0_09.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y8zpz9ij30zk0k0dmf.jpg)
![0_10.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76y9403wbj30zk0k0jvl.jpg)
![0_11.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ya2kygtj30zk0k0my4.jpg)
![0_12.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ya7p4osj30zk0k0tdu.jpg)
![0_13.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yad65ttj30zk0k0wis.jpg)
![0_14.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yaj73k7j30zk0k043f.jpg)
![0_15.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yan4si0j30zk0k077x.jpg)
![0_16.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yaryzvvj30zk0k00wc.jpg)
![0_17.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yavwynvj30zk0k0wf9.jpg)
![0_18.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yb8ppo6j30zk0k0766.jpg)
![0_19.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ybcvxbyj30zk0k0tbm.jpg)
![0_20.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ybicmgrj30zk0k0ju8.jpg)
![0_21.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ybndq1lj30zk0k0gmm.jpg)
![0_22.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ybsp16gj30zk0k0wkr.jpg)
![0_23.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ybwrvnbj30zk0k079l.jpg)
![0_24.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yc2gng6j30zk0k0t9k.jpg)
![0_25.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76yc72fm1j30zk0k041y.jpg)
![0_26.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ycc4ns5j30zk0k0dgp.jpg)
![0_27.jpg](https://ws1.sinaimg.cn/large/006CCxP6gy1g76ycgbgr7j30zk0k0n0p.jpg)

[ppt下载](https://pan.baidu.com/s/1GdlfsaWsnrKRaCYSpeBU2A)  密码: qhj9

这篇讲稿大概花了我近半个月时间，这期间阅读论文、推导公式、查阅wiki等等，都是对自己的锻炼。也经过这样一段时间，自己读论文的速度也有了提升，收获很大。

除此之外，今天读完一篇论文《GRAPH ATTENTION NETWORKS 》，主要讲解的是将Attention机制应用至GNN上，对以前的一些方法来讲，有显著的提升。最近也会分享出来。
___
~~最近感冒了，难受.......~~