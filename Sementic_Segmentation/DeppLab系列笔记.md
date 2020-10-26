[TOC](DeepLab系列笔记)

# 综述

## 蒋博笔记

> 语义分割是对图像中每个像素进行划分，本质上仍旧是一种逐像素的分类问题。语义分割的方法可以抽象为：全卷积CNN对图像进行特征提取，然后融合浅层特征进行上采样将score map恢复到输入大小。其中涉及到的几个重要的部分为：CNN backbone、特征融合策略、上采样策略。另外，与图像分类不同的是，单看每个像素都是一个孤立点，要想分割效果好就要充分利用当前像素点周围的特征信息。语义分割的论文始终都是在围绕这四个点进行不断迭代改进，只是所选择的思路有所不同。比如对于“pool层在扩大感受野的同时会损失空间信息”这个问题，FCN采用与浅层特征相融合的方式解决，而DeepLab系列的策略则是用空洞卷积代替pool层，在不损失空间信息的前提下增大卷积的感受野。-----------------------电子科技大学 蒋承知

# FCN

![](https://raw.githubusercontent.com/UESTC-Liuxin/paperNote/main/img/image-20200921144144243.png)

这是基于深度学习的端到端的语义分割方法的师祖了，利用反卷积替代全连接层，直接从特征图上得到分割图。

- [反卷积解释](https://www.zhihu.com/question/48279880)

- [pytorch中反卷积的实现](https://www.cnblogs.com/kk17/p/10111768.html)

> **1.1 原理**
> 对于普通的CNN网络，将最后的全连接层变成卷积层得到相应的feature map而不是特征向量（此时的feature map空间大小为输入的1/32）。然后通过1*1卷积将通道数规范到与类别数相等，通过反卷积将feature map的空间尺寸直接一步扩大到与输入相等（FCN-32s）得到最终的输出。
> 仅仅只利用特征提取部分最后一层的feature map的话，会存在两个问题影响最终的分类性能：1、由于经过了5次最大池化此时feature map的大小仅为原始的1/32，损失了太多的空间信息，对于逐像素划分的任务来说此时的feature map是很粗糙的；2、从1/32的尺度一步恢复到原输入大小，空间尺寸的陡增其实也是不利于信息的恢复的。
> 高层网络损失的信息在低网络层中还存在，所以在反卷积的过程中逐步与低层（Pool3、Pool4）的feature map进行结合，一方面融合所需的空间信息，另一方面也以一个适宜的速率恢复空间大小（×2、×2、×8），得到最终的输出（FCN-16s、FCN-8s）。
> **1.2 关于改进的思考**
> (1) 从网络backbone的角度：论文中的网络backbone基于AlexNet，在FCN的算法框架下对于更复杂的场景将backbone替换成更深的网络，特征提取的效果会更好，并且用于融合的低层feature map选择会更加灵活多样。
> (2) 从损失函数的角度：FCN的原始损失是逐像素的分类损失，将原始的softmax loss换成引入间隔之后的大间隔损失应该也会有所提升，参考sphereface、cosface、arcface等。
> **1.3 为什么全卷积可以？**
> CVPR2016的Class Activation Mapping说明了：CNN中其实不仅隐含了关于输入的特征信息，同时也包含了物体的位置信息，只是全连接层把这种位置信息破坏掉了。全卷积结构保留了图像中的位置信息，结合提取出的特征从而得已对不同位置像素的类别做出正确的判断。

- pipeline

  ![](https://raw.githubusercontent.com/UESTC-Liuxin/paperNote/main/img/image-20200922153128928.png)
  
  

# SegNet

# U-Net

 

# PSPnet

# DeepLabV1

论文：[Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/pdf/1412.7062.pdf)

## 蒋博笔记

> ***4.1.1 DeepLab V1 (ICLR 2015)***
>
> V1采用backbone为VGG16的全卷积网络，通过CNN得到feature map并通过双线性插值恢复到原图大小得到coarse output，最后用CRF得到精细输出。V1的改进：
>
> (1) **空洞卷积**。FCN存在“多次下采样导致空间信息损失过多不利于稠密预测”的问题。但去掉maxpool层又会减小网络的感受野，V1选择了一种和FCN不同的解决思路，通过将网络最后两层maxpool层去掉，在相应的网络层引入空洞卷积在更多保留空间信息的同时对网络的感受野不产生影响，通过空洞卷积的引入最终的特征图大小从原输入的1/32变为1/8，特征密度也更大了。
>
> (2) **条件随机场CRF*****。***V1中运用CRF对CNN输出的coarse output进行精细化调整。CRF将coarse output建模为一个全连接的无向图模型，每个像素代表一个节点，边代表节点间的关系。CRF就相当于用每个节点的全局关系进行调整（个人理解类似于non-local的作用）。
>
> (3) **上采样策略不同。**不同于FCN中采用的反卷积以及SegNet中采用的pool indices策略。V1中直接采用**双线性插值**(×8)将feature map恢复到输入大小。
>
> (4) **多尺度特征融合。**V1中同样也融合了低网络层的特征，将网络前四个max pool的feature map提取出来(每个featuremap后接3\*3卷积、1\*1卷积，输出128通道的feature map)与主干网络输出concate起来之后一起送入最终的分类层中。

- 反卷积（[反卷积理解](https://www.jianshu.com/p/f743bd9041b3)）

- CRF条件随机场

  - **条件随机场(Conditional Random Fields, 以下简称CRF)**是给定一组输入序列条件下另一组输出序列的条件概率分布模。
  - **随机场**是由若干个位置组成的整体，当给每一个位置中按照某种分布随机赋予一个值之后，其全体就叫做随机场。
  - **马尔科夫随机场**是随机场的特例，他假设随机场中某一个位置的复制仅仅与和它相邻的位置的赋值有关，和与其不相邻的位置的赋值无关。
  - **条件随机场**是值对于观测值X，Y是给定X的条件输出，Y构成一个条件随机场。对于CRF，给出准确的数学语言描述：设X与Y是随机变量，P(Y|X)是给定X时Y的条件概率分布，若随机变量Y构成的是一个马尔科夫随机场，则称条件概率分布P(Y|X)是条件随机场。
  - **全连接CRF**是指对于$Y=[y_1,y_2,y_3,...y_N]$,有$P(y_i|X,Y_{i-1},Y_{i+1})=P(y_i|X,y_1,y_2,y_3,...y_N)$，其中$X=[x_1,x_2,x_n,...,x_N]$。
  - 具体的算法，没有搞懂，太数学了。可以参照https://zhuanlan.zhihu.com/p/53421692CRF条件随机场

- Pipeline

  - **backbone**：V1是基于VGG16的backbone上做的一个修改，首先看VGG16本身的网络结构：

  ![2](https://img-blog.csdnimg.cn/20181226102627804.png) ![7](https://img-blog.csdnimg.cn/20181226103237670.png)

  ​	改动如下：

  ​	在这基础上做的改进是使用 1\*1 的卷积层代替FC层，那么就变成了全卷积网络，输出得到的是得分图，也可以理解成概率图。将pool4和pool5的步长由2改	为1， 这样在原本FC7的位置，VGG网络总的步长由原来的32变为8（总步长=输入size/特征图size）。一般来说，池化层的步长为2，池化后输出大小变为	输入大小的一半。原VGG16模型有5次池化，缩小 $2^5=32$ 倍，修改后的VGG16有3次补步长为2的池化，缩小 $2^3 = 8 $ 倍，两次步长为1的池化，输出大小	基本不变，所以说VGG网络总的步长由原来的32变为8。并且在Pool4之后的三个卷积层利用了空洞卷积进行感受野的扩大。并在FC6（改为了1*1卷积）的	特征图上应用了rate=4的空洞卷积。

  - 双线性插值进行图像空间尺寸的恢复，根据上面的输出，对$\frac{1}{32}$的特征图利用反卷积的到与输入原图相同的大小。
- 多尺度融合，==取原图与第一、第二、第三、第四池化层的输出，进行两层的卷积（3*3卷积、128通道；1\*1卷积、128通道），然后结合输出，进行上采样并concate后进行1\*1卷积。==
  - 输入CRF，得到最后的分割图。
  
  <img src="https://img-blog.csdn.net/20180331144458490?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3podXplbWluNDU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="img" style="zoom:80%;" />

# DeepLabV2

论文： [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)

## 蒋博笔记 

> **4.1.2 DeepLab V2 (arXiv 2016、TPAMI 2017)**
>
> 相比于V1，V2的算法Pipeline没有太多的变化，在保留CRF后处理的基础上，区别（改进）在于：
>
> (1) **backbone升级**。V2的backbone从VGG16升级为ResNet-101，网络的表达能力更强；
>
> (2) **ASPP(Atrous Spatial Pyramid Pooling)模块**。V1中引入了空洞卷积但没有对其扩张率做太多的调整，空洞卷积的扩张率对应了不同的尺度和感受野。针对“图像中同一类物体存在不同的尺度”这一特点，V2中提出ASPP模块，对同一特征图并行地用多个不同扩张率（6，12，18，24）的空洞卷积进行处理然后进行特征融合。

- ASPP模块（atrous spatial pyramid pooling:空洞空间金字塔池化）

  由于DeepLabV1中的空洞率调整不大，所以在针对于“图像同一类物体存在不同尺度”这个问题上，没有得到很好解决，因此提出ASPP，并行地使用多个不同rate的空洞卷积实现多个感受野，得到多个尺度的特征的捕捉。

  <img src="https://img-blog.csdnimg.cn/20190212123514332.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxOTk3NjI1,size_16,color_FFFFFF,t_70" alt="img" style="zoom: 80%;" />

  具体的pipeline表示为：

  ![](https://raw.githubusercontent.com/UESTC-Liuxin/paperNote/main/img/image-20200924163522807.png)

  具体描述为(以全局stride=16为例)：

  - 将图片输入resnet101：

    - 经stride=2的7\*7卷积和3\*3卷积之后，得到一张$\frac{1}{4}w$的特征图；

    - 将得到的特征图输入4个残差模块组（分别包含3,4,23,3）个残差模块，其中前三个残差模块组中，~~每个组的第一个残差模块的3*3卷积的stride分别为{1,2,2}~~，==在v2版本是用的池化，v3才改为了stride=2的空洞卷积==，并且空洞率为{1,1,1}，通道输出分别为{64,128,256}。因此经过三个残差模块组后得到$\frac{1}{16}w * \frac{1}{16}h * 256$的特征图。
    - 在进入第四个残差模块组后，其包含的三个模块的空洞率分别为{2,4,8},最后得到一个比较$\frac{1}{16}w * \frac{1}{16}h * 512$的特征图。

  - 将特征图输入ASPP：

    - ASPP的四个卷积核大小以及空洞率分别为{1,3,3,3}和{6,12,18，24}，输出通道均为256。
    - 对4张特征图进行contact得到，256*4的特征图

  - 进入Decoder

    - 卷积后得到channel=classnum的输出。
    - 双线性插值得到与原图尺寸相同的分割图

  - 输入CRF得到结果。

    

# DeepLabV3

论文：[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)

## 蒋博笔记

> 4.1.3 DeepLab V3 (arXiv 2017)
> V3主要做了三件事：
> (1) ASPP模块进行了改进。V3中提到虽然不同扩张率的空洞卷积可以得到不同尺度的特征信息，但是随着扩张率的增大，kernel所能覆盖到的有效特征值越来越少，极端情况下会退化成1×1卷积。所以V3对ASPP的支路做了变化：用1×1卷积将rate=24的支路替换掉，同时为了结合图像全局信息还新增了全局平均池化支路，并运用双线性插值恢复到所需的尺寸。另外，ASPP还加入了BN层帮助训练。
> (2) 对于空洞卷积模块，文章给出了一种串行的网络设计使得网络可以更深；
> (3) 去掉了CRF后处理模块，DCNN的输出已经够好，后处理意义不大了。

蒋博说得已经比较清楚了。

- 级联结构的调整

  将原本deeplabv2中ASPP增加一张Image的全局池化。

  ![img](https://img-blog.csdnimg.cn/20190212123514353.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIxOTk3NjI1,size_16,color_FFFFFF,t_70)

  

# DeepLabV3+

## 蒋博笔记 

论文：[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

> V3+基于V3的改进和结构，主要的区别（改进）在于：
> (1) backbone升级。从原本的ResNet101改为了Xception，并且将深度可分离卷积应用到了ASPP模块中，所有的pool层都用；
> (2) 替换Pool层。V3+将Xception的结构也做了一定的改进，其中最重要的就是将所有的Pool层都用stride=2的深度可分离卷积进行了替换；
> (3) 上采样策略精细化。前三板的最终输出都是直接通过双线性插值直接恢复到输入大小。V3+中在双线性插值的基础上借鉴了Encoder-Decoder的结构：首先将特征图双线性插值上采样4倍，与浅层特征空间大小一致的特征图（ResNet中为Conv2，Xception中为第二个stride=2的输出）进行结合，这种策略能够得到更加准确的物体轮廓信息。特征融合的过程为：浅层特征先经过1×1卷积缩减通道数，然后与4倍上采样的特征图concate之后经过3×3卷积融合，最后再进行一次4倍的双线性插值得到最终的输出。

- Xception(论文：https://arxiv.org/pdf/1610.02357.pdf)

  - 基于对Inception-v3的另一种改进。

  - Xception的提出是基于：通道之间的相关性与空间相关性需要分开处理。采用 Separable Convolution（**极致的 Inception** 模块）来替换原来 Inception-v3中的卷积操作。

  - 深度可分离卷积

    深度可分离卷积就是在卷积过程中，将被卷积特征图的channel分为若干个group，对每个group应用不同的卷积核（极**端情况就是对每一个channel都应应用不同的卷积核，最后进行对通道进行concatenate**）。

    <img src="https://img-blog.csdnimg.cn/20181206105614489.png" alt="img" style="zoom:80%;" />

  - Xception的卷积模板

    先进行普通卷积操作，再对 1 × 1 1×11×1 卷积后的每个channel分别进行 3 × 3 3×33×3 卷积操作，最后将结果 concatenate：

    <img src="https://img-blog.csdnimg.cn/20181206103219193.png" alt="img" style="zoom:50%;" />

  - Xception的具体网络结构

    Xception的结构主要是基于ResNet的，将里面的卷积层换成了每个通道的可分离卷积

    <img src="https://img-blog.csdnimg.cn/20181206104943105.png" alt="img" style="zoom:80%;" />

  - Xception主要的作用体现在减少了参数量

- pipeline

  <img src="https://upload-images.jianshu.io/upload_images/4688102-a2569e23d72df245.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="img" style="zoom: 80%;" />

- 增加了特征图的Low-Level Features

  经历了ASPP后，将Low-Level Features直接与ASPP 进行concatenate，经历3\*3卷积后，再双线性插值得到与原图大小相同的预测图。

# Deeplab总结

> **4.2 各版本的优劣比较**
> **4.2.1 比较**
> (1) V1与FCN：相比之前FCN框架下的语义分割算法，DeepLab系列最重要的改进就在于引入空洞卷积机制来平衡网络感受野不断加大过程中丢失的空间信息，同时对于每一个点都增大了上下文信息。DeepLab中还引入了score map的后处理阶段，用CRF对网络上采样得到的输出运用概率图模型进一步的精炼。
> (2) V1与V2：用空洞卷积解决了空间信息损失的问题之后，语义分割还面临的一个问题是在同一个场景下同类物体可能会表现出不同的尺度。V2用ASPP模块提取多尺度的特征解决这个问题。
> (3) V2与V3：V3主要是对ASPP模块的完善。V2中用不同扩张率的空洞卷积来提取不同尺度的特征。但是V3中提出，当扩张率过大时卷积核所能覆盖到的有效特征会减少，极端情况下空洞卷积会退化成1×1卷积。针对这个问题V3删除掉了扩张率最大的支路用1×1卷积代替，同时增加了全局平均池化的支路引入全局信息。
> (4) V3与V3+：当主要的模块（空洞卷积、ASPP）都完善之后，V3+引入了更强大的网络backbone，并且对最后的网络上采样策略进行了完善，引入了Encoder-Decoder的结构，提高输入对轮廓信息的恢复能力。
> **4.2.2 个人想法**
> 1、DeepLab系列总结：DeepLab从V1到V3+始终是围绕着综述中的四点在改进，V2和V3过渡阶段提出并改进了ASPP，V3+将前面工作的精华都结合起来之后得到了最终的DeepLab形态。
> 2、V3/V3+为什么是stride从8变为16？ 空洞卷积需要在高分辨率的特征图上进行操作V1和V2的stride都为8，高分辨率的特征图会带来超大的计算量。但随着网络backbone的不断升级，ASPP的提出并完善，DeepLab的性能不断增强也就不需要这么高的特征图分辨率（同时在这个过程中CRF也变得没有意义），在保证一个比较好的性能的同时，将stride变为16能够加快计算。
> 3、为什么是Xception？ 类似2中的理由，空洞卷积需要的高分辨率特征会加大网络的计算量，Xception网络在性能提升的时候引入了深度可分离卷积可以进一步减小DeepLab的计算量。
> **4.3 关于改进的思考**
> (1) 用可变卷积替代空洞卷积。空洞卷积的初衷是为了在不过多损失空间信息的前提下保证感受野不变，但空洞卷积只能按照固定形状对感受野进行扩张。可变卷积在扩张卷积kernel感受野的时候还可以根据图像的信息进行形状的自适应，对于轮廓不规整的物体应该会有性能提升。
> (2) DeepLab的作者在之后的研究中还尝试了运用NAS来得到比更有效的ASPP架构。

