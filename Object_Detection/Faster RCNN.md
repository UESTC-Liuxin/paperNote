## Abstract

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations.

Advances like SPPnet [1] and Fast R-CNN [2] have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features—using the recently popular terminology of neural networks with “attention” mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model [3], our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.
**Index Terms**—Object Detection, Region Proposal, Convolutional Neural Network

摘要

最新的物体检测网络依靠region proposals算法来假设物体的位置。
   SPPnet [1]和Fast R-CNN [2]之类的进步减少了这些检测网络的运行时间，暴露了region proposals计算的瓶颈。 在这项工作中，我们引入了一个region proposals网络（RPN），该region proposals网络与检测网络共享全图像卷积特征，从而实现几乎免费的region proposals。  RPN是一个完全卷积的网络，可以同时预测每个位置的对象范围和对象得分。 对RPN进行了端到端的训练，以生成高质量的区域建proposal，Fast R-CNN将其用于检测。 通过共享RPN和Fast R-CNN的卷积特征，我们将RPN和Fast R-CNN进一步合并为一个网络——使用最近流行的带有“注意力”机制的神经网络术语，RPN组件告诉统一网络要关注的区域。 对于非常深的VGG-16模型[3]，我们的检测系统在GPU上的帧速率为5fps（包括所有步骤），同时在PASCAL VOC 2007、2012和2007上达到了最新的对象检测精度。  MS COCO数据集，每个图像仅包含300个建proposal。 在ILSVRC和COCO 2015比赛中，Faster R-CNN和RPN是在多个赛道上获得第一名的作品的基础。 **代码已公开提供**。

## 1 INTRODUCTION 

Recent advances in object detection are driven by the success of region proposal methods (e.g., [4]) and region-based convolutional neural networks (RCNNs) [5]. Although region-based CNNs were computationally expensive as originally developed in [5], their cost has been drastically reduced thanks to sharing convolutions across proposals [1], [2]. The latest incarnation, Fast R-CNN [2], achieves near real-time rates using very deep networks [3], when ignoring the time spent on region proposals. Now, proposals are the test-time computational bottleneck in state-of-the-art detection systems.

region proposal方法（例如[4]）和基于区域的卷积神经网络（RCNN）[5]的成功推动了对象检测的最新进展。 尽管如最初在[5]中开发的那样，基于区域的CNN在计算上很昂贵，但由于在proposal[1]，[2]之间共享卷积，因此其成本已大大降低。 最新的代表，Faster R-CNN [2]，当忽略region proposal花费的时间时，使用非常深的网络[3]实现了接近实时的速度。 现在，proposal是最先进的检测系统中的测试时间计算瓶颈。

Region proposal methods typically rely on inexpensive features and economical inference schemes.
**Selective Search** [4], one of the most popular methods, greedily merges **superpixels** based on engineered low-level features. Yet when compared to efficient detection networks [2], Selective Search is an order of magnitude slower, at 2 seconds per image in a CPU implementation. **EdgeBoxes** [6] currently provides the best tradeoff between proposal quality and speed, at 0.2 seconds per image. Nevertheless, the region proposal step still consumes as much running time as the detection network

Region proposal方法通常依赖于廉价的功能和经济的推理方案。选择性搜索[4]是最流行的方法之一，它基于工程化的底层特征贪婪地合并超像素。 然而，与高效的检测网络相比[2]，选择性搜索的速度要慢一个数量级，在CPU实现中每张图像2秒。  EdgeBoxes [6]当前提供建proposal质量和速度之间的最佳权衡，每张图像0.2秒。 尽管如此，region proposals步骤仍然消耗与检测网络一样多的运行时间。

One may note that fast region-based CNNs take advantage of GPUs, while the region proposal methods used in research are implemented on the CPU, making such runtime comparisons inequitable. An obvious way to accelerate proposal computation is to reimplement it for the GPU. This may be an effective engineering solution, but re-implementation ignores the down-stream detection network and therefore misses important opportunities for sharing computation.

可能有人注意到，基于区域的快速CNN充分利用了GPU的优势，而研究中使用的region proposals方法是在CPU上实现的，因此这种运行时比较是不公平的。 加速提proposal计算的一种明显方法是为GPU重新实现。 这可能是一种有效的工程解决方案，但是重新实现会忽略下游检测网络，因此会错过共享计算的重要机会.

In this paper, we show that an algorithmic change— computing proposals with a deep convolutional neural network—leads to an elegant and effective solution where proposal computation is nearly cost-free given the detection network’s computation. To this end, we introduce novel Region Proposal Networks (RPNs) that share convolutional layers with state-of-the-art object detection networks [1], [2]. By sharing convolutions at test-time, the marginal cost for computing proposals is small (e.g., 10ms per image).

在本文中，我们展示了算法的变化（使用深度卷积神经网络计算proposals），实现了一种优雅而有效的解决方案，考虑到检测网络的计算，proposals计算几乎是免费的。 为此，我们介绍了与最新的对象检测网络[1]，[2]共享卷积层的新颖的region proposals网络（RPN）。 通过在测试时共享卷积，计算建proposal的边际成本很小（例如，每张图片10毫秒）。

Our observation is that the convolutional feature maps used by region-based detectors, like Fast RCNN, can also be used for generating region proposals. On top of these convolutional features, we construct an RPN by adding a few additional convolutional layers that simultaneously regress region bounds and objectness scores at each location on a regular grid. The RPN is thus a kind of fully convolutional network (FCN) [7] and can be trained end-to-end specifically for the task for generating detection proposals.

我们的观察结果是，基于区域的检测器（如Fast RCNN）使用的卷积特征图也可用于生成区域建proposal。 在这些卷积特征之上，我们通过添加一些其他卷积层来构造RPN，这些卷积层同时回归(**regress**)规则网格上每个位置的区域边界和客观性得分。 因此，RPN是一种全卷积网络（FCN）[7]，可以专门针对生成检测建proposal的任务进行端到端训练。

RPNs are designed to efficiently predict region proposals with a wide range of scales and aspect ratios. In contrast to prevalent methods [8], [9], [1], [2] that use  multiple scaled images multiple filter sizes multiple references pyramids of images (Figure 1, a) or pyramids of filters (Figure 1, b), we introduce novel “anchor” boxes that serve as references at multiple scales and aspect ratios. Our scheme can be thought of as a pyramid of regression references (Figure 1, c), which avoids enumerating images or filters of multiple scales or aspect ratios. This model performs well when trained and tested using single-scale images and thus benefits running speed.

RPN旨在以各种比例尺和纵横比（**aspect ratio**）有效预测region proposals。 与使用多个缩放图像，多个滤镜大小，多个参考金字塔图像（图1，a）或滤镜金字塔（图1，b）的流行方法[8]，[9]，[1]，[2]相比， 我们介绍了新颖的“anchors”盒(**anchor box**)，它们可以作为多种比例和纵横比的参考。 我们的方案可以看作是回归参考的金字塔（图1，c），它避免了枚举具有多个比例或纵横比的图像或过滤器。 当使用单比例尺图像进行训练和测试时，此模型表现良好，从而提高了运行速度。

To unify RPNs with Fast R-CNN [2] object detection networks, we propose a training scheme that alternates between fine-tuning for the region proposal task and then fine-tuning for object detection, while keeping the proposals fixed. This scheme converges quickly and produces a unified network with convolutional features that are shared between both tasks.

为了将RPN与快速R-CNN [2]对象检测网络统一起来，我们提出了一种训练方案，该方案在对区域建proposal任务进行微调与对对象检测进行微调之间交替，同时保持建proposal不变。 该方案可以快速收敛，并生成具有卷积功能的统一网络，这两个任务之间可以共享该卷积功能。

We comprehensively evaluate our method on the PASCAL VOC detection benchmarks [11] where RPNs with Fast R-CNNs produce detection accuracy better than the strong baseline of Selective Search with Fast R-CNNs. Meanwhile, our method waives nearly all computational burdens of Selective Search at test-time—the effective running time for proposals is just 10 milliseconds. Using the expensive very deep models of [3], our detection method still has a frame rate of 5fps (including all steps) on a GPU, and thus is a practical object detection system in terms of both speed and accuracy. We also report results on the MS COCO dataset [12] and investigate the improvements on PASCAL VOC using the COCO data. Code has been made publicly available at https://github.com/shaoqingren/faster_ rcnn (in MATLAB) and https://github.com/ rbgirshick/py-faster-rcnn (in Python).

我们在PASCAL VOC检测基准[11]上全面（**comprehensively**）评估了我们的方法，其中具有快速R-CNN的RPN产生的检测精度要优于具有快速R-CNN的选择性搜索的成熟baseline。 同时，我们的方法在测试时几乎免除（**waive**）了“选择性搜索”的所有计算负担-提案的有效运行时间仅为10毫秒。 使用昂贵的非常深的模型[3]（这里指的是VGG），我们的检测方法在GPU上的帧速率仍然为5fps（包括所有步骤），因此在速度和准确性方面都是实用的对象检测系统。 我们还报告了MS COCO数据集的结果[12]，并使用COCO数据研究了PASCAL VOC的改进。 代码已在https://github.com/shaoqingren/faster_ rcnn（在MATLAB中）和https://github.com/rbgirshick/py-faster-rcnn（在Python中）中公开可用。

A preliminary version of this manuscript was published previously [10]. Since then, the frameworks of RPN and Faster R-CNN have been adopted and generalized to other methods, such as 3D object detection [13], part-based detection [14], instance segmentation [15], and image captioning [16]. Our fast and effective object detection system has also been built in commercial systems such as at Pinterests [17], with user engagement improvements reported.

该手稿（**manuscript** ）的初步（**preliminary** ）版本先前已发布[10]。 从那时起，RPN和Faster R-CNN的框架已被采用并推广到其他方法，例如3D对象检测[13]，基于零件的检测[14]，实例分割[15]和图像字幕[16]。  我们的快速有效的物体检测系统也已经建立在商业系统中，例如Pinterests [17]，据报道用户参与度有所提高。

In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the basis of several 1st-place entries [18] in the tracks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation. RPNs completely learn to propose regions from data, and thus can easily benefit from deeper and more expressive features (such as the 101-layer residual nets adopted in [18]). Faster R-CNN and RPN are also used by several other leading entries in these competitions2. These results suggest that our method is not only a cost-efficient solution for practical usage, but also an effective way of improving object detection accuracy.

在ILSVRC和COCO 2015竞赛中，Faster R-CNN和RPN是ImageNet检测，ImageNet本地化，COCO检测和COCO分割中几个第一名的基础[18]。  RPN完全学会了根据数据提proposal区域，因此可以轻松地从更深，更具表现力的功能（例如[18]中采用的101层残差网）中受益。 在这些比赛中，其他一些领先的参赛者也使用了Faster R-CNN和RPN2。 这些结果表明我们的方法不仅是一种实用的经济有效的解决方案，而且是提高物体检测精度的有效途径。

## 2 RELATED WORK 

**Object Proposals.** There is a large literature on object proposal methods. Comprehensive surveys and comparisons of object proposal methods can be found in [19], [20], [21]. Widely used object proposal methods include those based on grouping super-pixels (e.g., Selective Search [4], CPMC [22], MCG [23]) and those based on sliding windows (e.g., objectness in windows [24], EdgeBoxes [6]). Object proposal methods were adopted as external modules independent of the detectors (e.g., Selective Search [4] object detectors, RCNN [5], and Fast R-CNN [2]).

关于object proposal方法的文献很多。 可以在[19]，[20]，[21]中找到对object proposal方法的全面调查和比较。 广泛使用的object proposal方法包括基于超像素分组的方法（例如，选择性搜索[4]，CPMC [22]，MCG [23]）和基于滑动窗口的方法（例如，窗口中的物体[24]，EdgeBoxes [  6]）。 采用object proposal方法作为独立于检测器的外部模块（例如，选择性搜索[4]对象检测器，RCNN [5]和Fast R-CNN [2]）。

**Deep Networks for Object Detection.** 

The R-CNN method [5] trains CNNs end-to-end to classify the proposal regions into object categories or background.R-CNN mainly plays as a classifier, and it does not predict object bounds (except for refining by bounding box regression). Its accuracy depends on the performance of the region proposal module (see comparisons in [20]). Several papers have proposed ways of using deep networks for predicting object bounding boxes [25], [9], [26], [27]. In the OverFeat method [9], a fully-connected layer is trained to predict the box coordinates for the localization task that assumes a single object. The fully-connected layer is then turned into a convolutional layer for detecting multiple classspecific objects. The MultiBox methods [26], [27] generate region proposals from a network whose last fully-connected layer simultaneously predicts multiple class-agnostic boxes, generalizing the “singlebox” fashion of OverFeat. These class-agnostic boxes are used as proposals for R-CNN [5]. The MultiBox proposal network is applied on a single image crop or multiple large image crops (e.g., 224×224), in contrast to our fully convolutional scheme. MultiBox does not share features between the proposal and detection networks. We discuss OverFeat and MultiBox in more depth later in context with our method. Concurrent with our work, the DeepMask method [28] is developed for learning segmentation proposals

R-CNN方法[5]端到端训练CNN将提案区域分类为对象类别或背景。R-CNN主要充当分类器，并且不预测对象边界（通过边界框回归进行精炼除外）  ）。 它的准确性取决于区域提proposal模块的性能（请参见[20]中的比较）。 几篇论文提出了使用深度网络预测对象边界框的方法[25]，[9]（***是纽约大学Yann LeCun团队中Pierre Sermanet ，David Eigen和张翔等在13年撰写的一篇论文，本文改进了Alex-net，并用图像缩放和滑窗方法在test数据集上测试网络；提出了一种图像定位的方法；最后通过一个卷积网络来同时进行分类，定位和检测三个计算机视觉任务，并在ILSVRC2013中获得了很好的结果。***），[26]，[27]。 在OverFeat方法[9]中，训练了一个全连接层来预测假设单个对象的定位任务的框坐标。 然后将完全连接的层转换为卷积层，以检测多个类特定的对象。  MultiBox方法[26]，[27]从网络中生成区域提proposal，该网络的最后一个全连接层同时预测多个与类无关的盒子，从而概括了OverFeat的“单个盒子”方式。 这些与类无关的框用作R-CNN的建proposal[5]。 与我们的全卷积方案相比，MultiBox  proposal网络适用于单个图片作物或多个大图片作物（例如224×224）。  MultiBox在提proposal和检测网络之间不共享功能。 我们稍后将在我们的方法中更深入地讨论OverFeat和MultiBox。 与我们的工作同时，开发了DeepMask方法[28]用于学习分割proposals

Shared computation of convolutions [9], [1], [29], [7], [2] has been attracting increasing attention for efficient, yet accurate, visual recognition. The OverFeat paper [9] computes convolutional features from an image pyramid for classification, localization, and detection. Adaptively-sized pooling (SPP) [1] on shared convolutional feature maps is developed for efficient region-based object detection [1], [30] and semantic segmentation [29]. Fast R-CNN [2] enables end-to-end detector training on shared convolutional features and shows compelling accuracy and speed.

卷积的共享计算已吸引了越来越多的关注[9]，[1]，[29]，[7]，[2]，以进行有效而准确的视觉识别。  OverFeat论文[9]从图像金字塔计算卷积特征，以进行分类，定位和检测。 共享卷积特征图上的自适应大小池（SPP）[1]被开发用于有效的基于区域的对象检测[1]，[30]和语义分割[29]。Fast R-CNN [2]可以对共享卷积特征进行端到端检测器训练，并显示出令人信服(**compelling**)的准确性和速度。

## 3 FASTER R-CNN

Our object detection system, called Faster R-CNN, is composed of two modules. ==The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector [2] that uses the proposed regions.== The entire system is a single, unified network for object detection (Figure 2).Using the recently popular terminology of neural networks with ‘attention’ [31] mechanisms, the RPN module tells the Fast R-CNN module where to look.In Section 3.1 we introduce the designs and properties of the network for region proposal. In Section 3.2 we develop algorithms for training both modules with features shared.

我们的物体检测系统称为Faster R-CNN，它由两个模块组成。 第一个模块是提出区域的深层全卷积网络，第二个模块是使用提出的区域的Fast R-CNN检测器[2]。 整个系统是用于对象检测的单个统一网络（图2）。RPN模块使用最近流行的带有“注意力” [31]机制的神经网络术语，将Fast R-CNN模块告诉哪里。 在 3.1节，我们介绍用于区域提proposal的网络的设计和属性。 在第3.2节中，我们开发了用于训练具有共享功能的两个模块的算法。

**3.1 Region Proposal Networks** 

==A Region Proposal Network (RPN) takes an image (of any size) as input and outputs a set of rectangular object proposals, each with an objectness score.== We model this process with a fully convolutional network [7], which we describe in this section. Because our ultimate goal is to share computation with a Fast R-CNN object detection network [2], we assume that both nets share a common set of convolutional layers. In our experiments, we investigate the Zeiler and Fergus model [32] (ZF), which has 5 shareable convolutional layers and the Simonyan and Zisserman model [3] (VGG-16), which has 13 shareable convolutional layers.

区域投标网络（RPN）接收（任意大小的）图像作为输入，并输出一组矩形的目标投标，每个目标投标都有一个客观评分。 我们使用全卷积网络[7]对此过程进行建模，这将在本节中进行描述。 因为我们的最终目标是与快速R-CNN对象检测网络共享计算[2]，所以我们假设两个网络共享一组共同的卷积层。 在我们的实验中，我们研究了具有5个可共享卷积层的Zeiler和Fergus模型[32]（ZF），以及具有13个可共享卷积层的Simonyan和Zisserman模型[3]（VGG-16）。

==To generate region proposals, we slide a small network over the convolutional feature map output by the last shared convolutional layer.==This small network takes as input an n × n spatial window of the input convolutional feature map. Each sliding window is mapped to a lower-dimensional feature (256-d for ZF and 512-d for VGG, with ReLU [33] following). This feature is fed into two sibling fully connected layers—a box-regression layer (reg) and a box-classification layer (cls). We use n = 3 in this paper, noting that the effective receptive field on the input image is large (171 and 228 pixels for ZF and VGG, respectively). This mini-network is illustrated at a single position in Figure 3 (left). Note that because the mini-network operates in a sliding-window fashion, the fully-connected layers are shared across all spatial locations. This architecture is naturally implemented with an n×n convolutional layer followed by two sibling 1 × 1 convolutional layers (for reg and cls, respectively).

为了生成区域建proposal，我们在最后共享的卷积层输出的卷积特征图上滑动一个小型网络。 这个小网络将输入卷积特征图的n×n空间窗口作为输入。 每个滑动窗口都映射到一个较低维的特征（ZF为256-d，VGG为512-d，后面是ReLU [33]）。 此功能被馈入两个同级的全连接层-框回归层（reg）和框分类层（cls）。 我们在本文中使用n = 3，注意输入图像上的有效接收场很大（ZF和VGG分别为171和228像素）。 在图3的单个位置（左）显示了此微型网络。 请注意，由于微型网络以滑动窗口的方式运行，因此全连接层将在所有空间位置上共享。 这种体系结构自然是由n×n卷积层和两个同级1×1卷积层（分别用于reg和cls）实现的。

滑动窗口用的3X3卷积。

**3.1.1 Anchors**

At each sliding-window location, we simultaneously predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as k. So the reg layer has 4k outputs encoding the coordinates of k boxes, and the cls layer outputs 2k scores that estimate probability of object or not object for each proposal4. The k proposals are parameterized relative to k reference boxes, which we call anchors. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio (Figure 3, left). By default we use 3 scales and 3 aspect ratios, yielding k = 9 anchors at each sliding position. For a convolutional feature map of a size W × H (typically ∼2,400), there are W Hk anchors in total.

![这里写图片描述](https://img-blog.csdn.net/20171012142058650?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTGluX3hpYW95aQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

在每个滑动窗口位置，我们同时预测多个区域提proposal，其中每个位置的最大可能提proposal数目表示为k。 因此，reg层具有4k个输出，该输出对k个框的坐标进行编码，而cls层输出2k个分数（每个anchor有前景和背景两个类别的置信度分数），这些分数估计每个提案的目标或非目标的概率4。 相对于k个参考框（称为anchors），对k个proposals进行了参数化。 anchors点位于相关滑动窗口的中心，并与比例和长宽比相关（图3，左）。 默认情况下，我们使用3个比例和3个纵横比，在每个滑动位置产生k = 9个anchors。 对于大小为W×H（通常约为2400）的卷积特征图，总共有WHk个anchors点。

> ==（对于滑动窗口，作者就计算这个滑动窗口的中心点所对应的原始图片的中心点。对于每个3x3窗口，作者假定它来自9（论文中的k）种不同原始区域的池化，但是这些池化在原始图片中的中心点，都完全一样。这个中心点，就是刚才提到的，3x3窗口中心点所对应的原始图片中的中心点。如此一来，在每个窗口位置，我们都可以根据9个不同长宽比例、不同面积的anchor，逆向推导出它所对应的原始图片中的一个区域，这个区域的尺寸以及坐标，都是已知的。而这个区域，就是我们想要的 proposal。所以我们通过滑动窗口和anchor，成功得到了 anchors 个原始图片的proposal。接下来，每个proposal我们只输出6个参数：每个 proposal 和 ground truth 进行比较得到的前景概率和背景概率(2个参数：分别包括了前景和背景两个类别的置信度分数）（对应图上的 cls_score）；由于每个 proposal 和 ground truth 位置及尺寸上的差异，从 proposal 通过平移放缩得到 ground truth 需要的4个平移放缩参数（对应图上的 bbox_pred））==  

**Translation-Invariant Anchors** 

An important property of our approach is that it is translation invariant, both in terms of the anchors and the functions that compute proposals relative to the anchors. If one translates an object in an image, the proposal should translate and the same function should be able to predict the proposal in either location. This translation-invariant property is guaranteed by our method5. As a comparison, the MultiBox method [27] uses k-means to generate 800 anchors, which are not translation invariant. So MultiBox does not guarantee that the same proposal is generated if an object is translated.

我们方法的一个重要特性是，就anchors和计算相对于anchors的proposals 的功能而言，它是变换不变的。 如果一个人变换了图像中的一个对象，则该proposal 应进行变换，并且相同的功能应能够在任一位置预测该proposals 。 此平移不变属性由我们的方法5保证。 作为比较，MultiBox方法[27]使用k均值生成800个anchors点，这些anchors点不是平移不变的。 因此，如果变换了对象，则MultiBox不能保证生成相同的建proposal.

The translation-invariant property also reduces the model size. MultiBox has a (4 + 1) × 800-dimensional fully-connected output layer, whereas our method has a (4 + 2) × 9-dimensional convolutional output layer in the case of k = 9 anchors. As a result, our output layer has 2.8 × 10^4 parameters (512 × (4 + 2) × 9 for VGG-16), two orders of magnitude fewer than MultiBox’s output layer that has 6.1 × 10^6 parameters (1536 × (4 + 1) × 800 for GoogleNet [34] in MultiBox [27]). If considering the feature projection layers, our proposal layers still have an order of magnitude fewer parameters than MultiBox6. We expect our method to have less risk of overfitting on small datasets, like PASCAL VOC.

平移不变属性还减小了模型大小。  MultiBox具有（4 +1）×800维的全连接输出层，而在k = 9 anchor的情况下，我们的方法具有（4 + 2）×9维的卷积输出层。 结果，我们的输出层具有2.8×10 ^ 4个参数（VGG-16为512×4 + 2×9），比具有6.1×10 ^ 6参数（1536× 4 + 1×800（对于MultiBox [27]中的GoogleNet [34]）。 如果考虑特征投影层，我们的建proposal层的参数仍然比MultiBox6少一个数量级。 我们希望我们的方法在较小的数据集（如PASCAL VOC）上过拟合的风险较小。

**Multi-Scale Anchors as Regression References** 

Our design of anchors presents a novel scheme for addressing multiple scales (and aspect ratios). As shown in Figure 1, there have been two popular ways for multi-scale predictions. The first way is based on image/feature pyramids, e.g., in DPM [8] and CNNbased methods [9], [1], [2]. The images are resized at multiple scales, and feature maps (HOG [8] or deep convolutional features [9], [1], [2]) are computed for each scale (Figure 1(a)). This way is often useful but is time-consuming. The second way is to use sliding windows of multiple scales (and/or aspect ratios) on the feature maps. For example, in DPM [8], models of different aspect ratios are trained separately using different filter sizes (such as 5×7 and 7×5). If this way is used to address multiple scales, it can be thought of as a “pyramid of filters” (Figure 1(b)). The second way is usually adopted jointly with the first way [8].

我们的anchors设计提出了一种解决多种比例（和纵横比）的新颖方案。 如图1所示，有两种流行的多尺度预测方法。 第一种方法是基于图像/特征金字塔的，例如在DPM [8]和基于CNN的方法[9]，[1]，[2]中。 图像在多个比例上调整大小，并为每个比例计算特征图（HOG [8]或深度卷积特征[9]，[1]，[2]）（图1（a））。 这种方法通常有用但很费时。 第二种方法是在特征图上使用多个比例（和/或纵横比）的滑动窗口。 例如，在DPM [8]中，使用不同的滤镜大小（例如5×7和7×5）分别训练不同长宽比的模型。 如果使用这种方法处理多个尺度，则可以将其视为“过滤器金字塔”（图1（b））。 第二种方法通常与第一种方法一起使用[8]。

==这里涉及到的9个尺寸来源于3种横纵比3种面积的自由组合，用于取代图像金字塔的策略== 。

As a comparison, our anchor-based method is built on a pyramid of anchors, which is more cost-efficient.Our method classifies and regresses bounding boxes with reference to anchor boxes of multiple scales and aspect ratios. It only relies on images and feature maps of a single scale, and uses filters (sliding windows on the feature map) of a single size. We show by experiments the effects of this scheme for addressing multiple scales and sizes (Table 8).

Because of this multi-scale design based on anchors, we can simply use the convolutional features computed on a single-scale image, as is also done by the Fast R-CNN detector [2]. The design of multiscale anchors is a key component for sharing features without extra cost for addressing scales.

相比之下，我们的基于anchors的方法是基于anchors的金字塔构建的，这种方法更具成本效益。 我们的方法参照多个比例和纵横比的anchors框对边界框进行分类和回归。 它仅依赖单一比例的图像和特征图，并使用单一尺寸的过滤器（特征图上的滑动窗口）。 我们通过实验证明了该方案对解决多种尺度和尺寸的影响（表8）。
   由于基于anchors的这种多尺度设计，我们可以简单地使用在单尺度图像上计算出的卷积特征，就像快速R-CNN检测器所做的一样[2]。 多尺度anchors的设计是共享要素而无需花费额外成本来解决尺度的关键组成部分。

**3.1.2 Loss Function**

For training RPNs, we assign a binary class label (of being an object or not) to each anchor. We assign a positive label to two kinds of anchors: (i) the anchor/anchors with the highest Intersection-overUnion (IoU) overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box. Note that a single ground-truth box may assign positive labels to multiple anchors.Usually the second condition is sufficient to determine the positive samples; but we still adopt the first condition for the reason that in some rare cases the second condition may find no positive sample. We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes.Anchors that are neither positive nor negative do not contribute to the training objective.

为了训练RPN，我们为每个anchor分配一个二进制类标签（无论是不是对象）。 我们为两种anchor定一个正标签：（i）具有最高Intersection-overUnion（IoU）重叠的anchor点/anchor点与地面真格，或（ii）任何与ground-truth box具有大于0.7的IoU重叠点的anchor点。 请注意，单个ground-truth box可以为多个anchor点分配正例标签。通常，第二个条件足以确定正例样本。 但是我们仍然采用第一个条件，因为在极少数情况下，第二个条件可能找不到阳性样本。 如果所有地面真相盒子的IoU值均低于0.3，我们会给非阳性anchor定一个否定标签。既不是阳性也不是阴性的anchor都不有助于训练目标。

With these definitions, we minimize an objective function following the multi-task loss in Fast R-CNN [2]. Our loss function for an image is defined as:

利用这些定义，我们在快速R-CNN [2]中将多任务损失之后的目标函数减至最小。 我们对图像的损失函数定义为：



$$
\begin{aligned}
L\left(\left\{p_{i}\right\},\left\{t_{i}\right\}\right) &=\frac{1}{N_{c l s}} \sum_{i} L_{c l s}\left(p_{i}, p_{i}^{*}\right) \\
+\lambda & \frac{1}{N_{r e g}} \sum_{i} p_{i}^{*} L_{r e g}\left(t_{i}, t_{i}^{*}\right)
\end{aligned}
$$
在此，i是mini-batch中的anchor的索引，pi是anchor点i作为对象的预测概率。 如果anchor为正，则真实标签$p^*_i$为1，如果anchor点为负，则为0。  ti是代表预测边界框的4个参数化坐标的向量，而$t^∗_i$是与正anchor关联的地面真值框的参数化坐标。
   分类损失$L_{cls}$是两个类别（对象与非对象）之间的对数损失。 对于回归损失，我们使用$L_{r e g}\left(t_{i}, t_{i}^{*}\right)=R\left(t_{i}-t_{i}^{*}\right)$，其中R是在[2]中定义的稳健损失函数（平滑L1）。 $p_{i}^{*} L_{r e g}$表示仅对正anchor点$\left(p_{i}^{*}=1\right)$激活回归损失，否则对回归损失禁用$\left(p_{i}^{*}=0\right)$。   cls和reg层的输出分别由${p_i}$和${t_i}$组成。
   两项通过$N_{cls}$和$N_{reg}$归一化，并通过平衡参数λ加权。 在我们当前的实现中（如发布的代码中一样），等式（1）中的cls项通过小批量大小（即$N_{cls}=256$）进行归一化，而reg项通过anchor点位置的数量（即，$N_{r e g} \sim 2,400$）。   默认情况下，我们将$\lambda$设置为10，因此cls和reg项的权重大致相等。 我们通过实验表明，结果对宽范围内的λ值不敏感（表9）。 我们还注意到，上面的标准化不是必需的，可以简化。
$$
\begin{aligned}
t_{\mathrm{x}} &=\left(x-x_{\mathrm{a}}\right) / w_{\mathrm{a}}, \quad t_{\mathrm{y}}=\left(y-y_{\mathrm{a}}\right) / h_{\mathrm{a}} \\
t_{\mathrm{w}} &=\log \left(w / w_{\mathrm{a}}\right), \quad t_{\mathrm{h}}=\log \left(h / h_{\mathrm{a}}\right) \\
t_{\mathrm{x}}^{*} &=\left(x^{*}-x_{\mathrm{a}}\right) / w_{\mathrm{a}}, \quad t_{\mathrm{y}}^{*}=\left(y^{*}-y_{\mathrm{a}}\right) / h_{\mathrm{a}} \\
t_{\mathrm{w}}^{*} &=\log \left(w^{*} / w_{\mathrm{a}}\right), \quad t_{\mathrm{h}}^{*}=\log \left(h^{*} / h_{\mathrm{a}}\right)
\end{aligned}
$$
其中x，y，w和h表示框的中心坐标及其宽度和高度。变量x，$x_a$和$x^*$分别用于预测框，anchor生成的框和真实框。

==关于bbox的回归，作者采用了图示这样的方法，P代表实际预测的predicted bbox，也就是anchor直接生成的框，G代表真实的ground truth bbox，而作者希望通过学习一种运算，把P转换到$\hat{G}$,把它当作最后的一簇额输出，$\hat{G}\approx G$  ,实际上就是去拟合G。其中这种运算其实并不神秘，就是平移和伸缩。在学习过程中，学习的参数实际上就是$t_x,t_w$等，优化方法为：利用anchor生成的proposal的bbox去与ground truth bbox做比较，得到真实需要的${t_x^*}$等，损失的计算就是$L_{r e g}\left(t_{i}, t_{i}^{*}\right)$。从而不断优化$t_x$==

![](https://img-blog.csdn.net/20170831205020797?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvemlqaW4wODAyMDM0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

Nevertheless, our method achieves bounding-box regression by a different manner from previous RoIbased (Region of Interest) methods [1], [2]. In [1], [2], bounding-box regression is performed on features pooled from arbitrarily sized RoIs, and the regression weights are shared by all region sizes. In our formulation, the features used for regression are of the same spatial size (3 × 3) on the feature maps. To account for varying sizes, a set of k bounding-box regressors are learned. Each regressor is responsible for one scale and one aspect ratio, and the k regressors do not share weights. As such, it is still possible to predict boxes of various sizes even though the features are of a fixed size/scale, thanks to the design of anchors.

然而，我们的方法通过与以前基于RoI的（感兴趣区域）方法[1]，[2]不同的方式实现包围盒回归。 在[1]，[2]中，对从任意大小的RoI合并的特征执行边界框回归，并且回归权重由所有区域大小共享。 在我们的公式中，用于回归的特征在特征图上具有相同的空间大小（3×3）。 为了说明变化的大小，学习了一组k个边界框回归器。 每个回归器负责一个比例和一个长宽比，而k个回归器不共享权重。 这样，由于anchor的设计，即使特征具有固定的大小/比例，仍然可以预测各种大小的盒子。

**3.1.3 Training RPNs** 

The RPN can be trained end-to-end by backpropagation and stochastic gradient descent (SGD) [35]. We follow the “image-centric” sampling strategy from [2] to train this network. Each mini-batch arises from a single image that contains many positive and negative example anchors. It is possible to optimize for the loss functions of all anchors, but this will bias towards negative samples as they are dominate.Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1. If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones.

可以通过反向传播和随机梯度下降（SGD）端对端地训练RPN [35]。 我们遵循[2]中的“以图像为中心”的采样策略来训练该网络。 每个微型批处理均来自包含多个正负示例锚的单个图像。 可以对所有anchor的损失函数进行优化，但这会偏向负样本，因为它们占主导地位，相反，我们在图像中随机采样256个anchor以计算微型批次的损失函数，其中正样本 负锚的比例最高为1：1。 如果图像中的正样本少于128个，则用负样本填充小批量。

We randomly initialize all new layers by drawing weights from a zero-mean Gaussian distribution with standard deviation 0.01. All other layers (i.e., the shared convolutional layers) are initialized by pretraining a model for ImageNet classification [36], as is standard practice [5]. We tune all layers of the ZF net, and conv3 1 and up for the VGG net to conserve memory [2]. We use a learning rate of 0.001 for 60k mini-batches, and 0.0001 for the next 20k mini-batches on the PASCAL VOC dataset. We use a momentum of 0.9 and a weight decay of 0.0005 [37].Our implementation uses Caffe [38].

我们通过从零均值高斯分布中提取权重（标准偏差为0.01）来随机初始化所有新层。 所有其他层（即共享卷积层）都通过预先训练ImageNet分类模型来初始化[36]，这是标准做法[5]。 我们调整ZF网络的所有层，并调整conv3 1以及VGG网络以节省内存[2]。 对于PASCAL VOC数据集，我们对==6万个小批量使用0.001的学习率，对接下来的20k小批量使用0.0001的学习率。 我们使用0.9的动量和0.0005的权重衰减[37]==。我们的实现使用Caffe [38]。

**3.2 Sharing Features for RPN and Fast R-CNN** 

Thus far we have described how to train a network for region proposal generation, without considering the region-based object detection CNN that will utilize these proposals. For the detection network, we adopt Fast R-CNN [2]. Next we describe algorithms that learn a unified network composed of RPN and Fast R-CNN with shared convolutional layers (Figure 2).

Both RPN and Fast R-CNN, trained independently, will modify their convolutional layers in different ways. We therefore need to develop a technique that allows for sharing convolutional layers between the two networks, rather than learning two separate networks. We discuss three ways for training networks with features shared:

 (i) Alternating training. In this solution, we first train RPN, and use the proposals to train Fast R-CNN.The network tuned by Fast R-CNN is then used to initialize RPN, and this process is iterated. This is the solution that is used in all experiments in this paper.

(ii) Approximate joint training. In this solution, the RPN and Fast R-CNN networks are merged into one network during training as in Figure 2. In each SGD iteration, the forward pass generates region proposals which are treated just like fixed, pre-computed proposals when training a Fast R-CNN detector. The backward propagation takes place as usual, where for the shared layers the backward propagated signals from both the RPN loss and the Fast R-CNN loss are combined. This solution is easy to implement. But this solution ignores the derivative w.r.t. the proposal boxes’ coordinates that are also network responses, so is approximate. In our experiments, we have empirically found this solver produces close results, yet reduces the training time by about 25-50% comparing with alternating training. This solver is included in our released Python code.

(iii) Non-approximate joint training. As discussed above, the bounding boxes predicted by RPN are also functions of the input. The RoI pooling layer [2] in Fast R-CNN accepts the convolutional features and also the predicted bounding boxes as input, so a theoretically valid backpropagation solver should also involve gradients w.r.t. the box coordinates. These gradients are ignored in the above approximate joint training. In a non-approximate joint training solution, we need an RoI pooling layer that is differentiable w.r.t. the box coordinates. This is a nontrivial problem and a solution can be given by an “RoI warping” layer as developed in [15], which is beyond the scope of this paper

到目前为止，我们已经描述了如何训练用于区域提proposal生成的网络，而没有考虑将利用这些proposals的基于区域的对象检测CNN。 对于检测网络，我们采用Fast R-CNN [2]。 接下来，我们描述学习具有RPN和Fast R-CNN并具有共享卷积层的统一网络的算法（图2）。
    RPN和Fast R-CNN均经过独立训练，将以不同方式修改其卷积层。 因此，我们需要开发一种技术，允许在两个网络之间共享卷积层，而不是学习两个单独的网络。 我们讨论了共享功能的三种训练网络的方法：

（i）交替训练。 在此解决方案中，我们首先训练RPN，然后使用proposal来训练Fast R-CNN，然后使用由Fast R-CNN调谐的网络初始化RPN，然后重复此过程。 这是本文所有实验中使用的解决方案。

（ii）近似联合训练。 在此解决方案中，如图2所示，在训练过程中将RPN和Fast R-CNN网络合并为一个网络。在每次SGD迭代中，前向传递都会生成区域proposal，在训练Fast R-CNN检测器时就像对待固定的，预先计算的proposal一样 。 反向传播照常进行，对于共享层，来自RPN损耗和快速R-CNN损耗的反向传播信号被组合在一起。 该解决方案易于实现。 但是此解决方案忽略了导数w.r.t. 提案框的坐标也是网络响应，因此是近似值。 在我们的实验中，我们凭经验发现此求解器产生的结果接近，但与交替训练相比，训练时间减少了约25-50％。 此求解器包含在我们发布的Python代码中。

（iii）非近似联合训练。 如上所述，RPN预测的边界框也是输入的函数。 快速R-CNN中的RoI合并层[2]接受卷积特征，并接受预测的边界框作为输入，因此，理论上有效的反向传播求解器也应包含梯度w.r.t。 框坐标。 这些梯度在上面的近似联合训练中被忽略。 在一个非近似的联合训练解决方案中，我们需要一个w.r.t. 框坐标。 这是一个不平凡的问题，可以通过[15]中开发的“ RoI翘曲”层来提供解决方案，这超出了本文的范围。

**4-Step Alternating Training.**

In this paper, we adopt a pragmatic 4-step training algorithm to learn shared features via alternating optimization. In the first step, we train the RPN as described in Section 3.1.3. This network is initialized with an ImageNet-pre-trained model and fine-tuned end-to-end for the region proposal task. In the second step, we train a separate detection network by Fast R-CNN using the proposals generated by the step-1 RPN. This detection network is also initialized by the ImageNet-pre-trained model. At this point the two networks do not share convolutional layers. In the third step, we use the detector network to initialize RPN training, but we fix the shared convolutional layers and only fine-tune the layers unique to RPN. Now the two networks share convolutional layers. Finally, keeping the shared convolutional layers fixed, we fine-tune the unique layers of Fast R-CNN. As such, both networks share the same convolutional layers and form a unified network. A similar alternating training can be run for more iterations, but we have observed negligible improvements.

在本文中，我们采用4步训练算法来通过交替优化学习共享特征。 第一步，我们按照3.1.3节所述训练RPN。 该网络使用ImageNet预训练的模型初始化，并针对区域建议任务端到端进行了微调。 在第二步中，我们使用步骤1 RPN生成的建议，通过Fast R-CNN训练一个单独的检测网络。 该检测网络也由ImageNet预训练模型初始化。 此时，两个网络不共享卷积层。 在第三步中，我们使用检测器网络初始化RPN训练，但是我们修复了共享卷积层，并且仅微调了RPN唯一的层。 现在，这两个网络共享卷积层。 最后，保持共享卷积层固定不变，我们对Fast R-CNN的唯一层进行微调。 这样，两个网络共享相同的卷积层并形成统一的网络。 可以进行类似的交替训练进行更多迭代，但是我们观察到的改进微不足道。

**3.3 Implementation Details**  

We train and test both region proposal and object detection networks on images of a single scale [1], [2].We re-scale the images such that their shorter side is s = 600 pixels [2]. Multi-scale feature extraction (using an image pyramid) may improve accuracy but does not exhibit a good speed-accuracy trade-off [2].
On the re-scaled images, the total stride for both ZF and VGG nets on the last convolutional layer is 16 pixels, and thus is ∼10 pixels on a typical PASCAL image before resizing (∼500×375). Even such a large stride provides good results, though accuracy may be further improved with a smaller stride.

For anchors, we use 3 scales with box areas of 1282, 2562, and 5122 pixels, and 3 aspect ratios of 1:1, 1:2, and 2:1. These hyper-parameters are not carefully chosen for a particular dataset, and we provide ablation experiments on their effects in the next section. As discussed, our solution does not need an image pyramid or filter pyramid to predict regions of multiple scales, saving considerable running time. Figure 3 (right) shows the capability of our method for a wide range of scales and aspect ratios. Table 1 shows the learned average proposal size for each anchor using the ZF net. We note that our algorithm allows predictions that are larger than the underlying receptive field.Such predictions are not impossible—one may still roughly infer the extent of an object if only the middle of the object is visible.

The anchor boxes that cross image boundaries need to be handled with care. During training, we ignore all cross-boundary anchors so they do not contribute to the loss. For a typical 1000 × 600 image, there will be roughly 20000 (≈ 60 × 40 × 9) anchors in total. With the cross-boundary anchors ignored, there are about 6000 anchors per image for training. If the boundary-crossing outliers are not ignored in training, they introduce large, difficult to correct error terms in the objective, and training does not converge. During testing, however, we still apply the fully convolutional RPN to the entire image. This may generate crossboundary proposal boxes, which we clip to the image boundary.

我们在单一比例尺的图像上训练和测试区域提议和目标检测网络[1]，[2]。我们重新缩放图像，使其短边为s = 600像素[2]。 ==多尺度特征提取（使用图像金字塔）可能会提高准确性==，但并不能表现出良好的速度精度折中[2]。
在重新缩放的图像上，最后一个卷积层上的ZF和VGG网络的总跨度为16像素，因此在调整大小之前，在典型的PASCAL图像上的总跨度为〜10像素（〜500×375）。 即使跨度较大，也可以提供良好的结果，尽管跨度较小时可以进一步提高精度。

 对于anchor，我们使用3个比例，框区域分别为1282、2562和5122像素，以及3个纵横比为1：1、1：2和2：1。 ==这些超参数不是为特定数据集精心选择的==，我们将在下一部分中提供有关其影响的消融实验。 如前所述，我们的解决方案不需要图像金字塔或滤镜金字塔即可预测多个尺度的区域，从而节省了可观的运行时间。 图3（右）显示了我们的方法在各种比例尺和纵横比下的功能。 表1显示了使用ZF网络为每个anchor学习的平均建议大小。 我们注意到，我们的算法所允许的预测要大于潜在的接受场。这种预测并非不可能-如果只有对象的中间可见，则仍可以粗略地推断出对象的范围。

 跨越图像边界的anchor box需要小心处理。 在训练期间，我们将忽略所有跨边界anchor ，因此它们不会造成损失。 对于典型的1000×600图像，总共将有大约20000（≈60×40×9）个anchor。 忽略跨边界anchor，每个图像大约有6000个anchor用于训练。 ==如果在训练中不忽略跨边界的异常值，则会在目标中引入较大且难以校正的误差项，并且训练不会收敛==。 但是，==在测试过程中，我们仍将全卷积RPN应用于整个图像。 这可能会生成跨边界建议框，我们会将其裁剪到图像边界。==

Some RPN proposals highly overlap with each other. To reduce redundancy, we adopt non-maximum suppression (NMS) on the proposal regions based on their cls scores. We fix the IoU threshold for NMS at 0.7, which leaves us about 2000 proposal regions per image. As we will show, NMS does not harm the ultimate detection accuracy, but substantially reduces the number of proposals. After NMS, we use the top-N ranked proposal regions for detection. In the following, we train Fast R-CNN using 2000 RPN proposals, but evaluate different numbers of proposals at test-time。

一些RPN proposals 彼此高度重叠。 为了减少冗余，我们根据提案区域的cls分数采用非最大抑制（NMS）。 我们将NMS的IoU阈值固定为0.7，这使得每个图像有大约2000个建议区域。 正如我们将显示的那样，NMS不会损害最终的检测准确性，但是会大大减少提议的数量。 在NMS之后，我们使用排名前N位的提案区域进行检测。 接下来，我们使用2000个RPN提案训练Fast R-CNN，但在测试时评估不同数量的提案。

## 4 EXPERIMENTS 

**4.1 Experiments on PASCAL VOC**

We comprehensively evaluate our method on the PASCAL VOC 2007 detection benchmark [11]. This dataset consists of about 5k trainval images and 5k test images over 20 object categories. We also provide results on the PASCAL VOC 2012 benchmark for a few models. For the ImageNet pre-trained network, we use the “fast” version of ZF net [32] that has 5 convolutional layers and 3 fully-connected layers, and the public VGG-16 model7 [3] that has 13 convolutional layers and 3 fully-connected layers. We primarily evaluate detection mean Average Precision (mAP), because this is the actual metric for object detection (rather than focusing on object proposal proxy metrics).

我们根据PASCAL VOC 2007检测基准[11]全面评估了我们的方法。 该数据集由大约20个对象类别的5k训练图像和5k测试图像组成。 我们还提供了一些型号的PASCAL VOC 2012基准测试结果。 对于ImageNet预训练网络，我们使用具有5个卷积层和3个完全连接层的“快速”版本的ZF net [32]，以及具有13个卷积层和3个公共VGG-16 model7 [3]。 完全连接的层。 我们主要评估检测平均平均精度（mAP），因为这是对象检测的实际指标（而不是关注对象建议代理指标）。

Table 2 (top) shows Fast R-CNN results when trained and tested using various region proposal methods. These results use the ZF net. For Selective Search (SS) [4], we generate about 2000 proposals by the “fast” mode. For EdgeBoxes (EB) [6], we generate the proposals by the default EB setting tuned for 0.7 IoU. SS has an mAP of 58.7% and EB has an mAP of 58.6% under the Fast R-CNN framework. RPN with Fast R-CNN achieves competitive results, with an mAP of 59.9% while using up to 300 proposals8.Using RPN yields a much faster detection system than using either SS or EB because of shared convolutional computations; the fewer proposals also reduce the region-wise fully-connected layers’ cost (Table 5).

表2（顶部）显示了使用各种区域建议方法进行训练和测试时的快速R-CNN结果。 这些结果使用ZF网络。 对于选择性搜索（SS）[4]，我们通过“快速”模式生成了大约2000个建议。 对于EdgeBoxes（EB）[6]，我们通过调整为0.7 IoU的默认EB设置生成建议。 在Fast R-CNN框架下，SS的mAP为58.7％，EB的mAP为58.6％。 具有快速R-CNN的RPN取得了竞争性结果，当使用多达300个建议时，mAP为59.9％8。由于共享卷积计算，与使用SS或EB相比，使用RPN产生的检测系统要快得多。 较少的提议也降低了区域级全连接层的成本（表5）。

**Ablation Experiments on RPN.** 

To investigate the behavior of RPNs as a proposal method, we conducted several ablation studies. First, we show the effect of sharing convolutional layers between the RPN and Fast R-CNN detection network. To do this, we stop after the second step in the 4-step training process.Using separate networks reduces the result slightly to 58.7% (RPN+ZF, unshared, Table 2). We observe that this is because in the third step when the detectortuned features are used to fine-tune the RPN, the proposal quality is improved.

Next, we disentangle the RPN’s influence on training the Fast R-CNN detection network. For this purpose, we train a Fast R-CNN model by using the 2000 SS proposals and ZF net. We fix this detector and evaluate the detection mAP by changing the proposal regions used at test-time. In these ablation experiments, the RPN does not share features with the detector.

Replacing SS with 300 RPN proposals at test-time leads to an mAP of 56.8%. The loss in mAP is because of the inconsistency between the training/testing proposals. This result serves as the baseline for the following comparisons.

为了研究RPNs作为提议方法的行为，我们进行了一些消融研究。 首先，我们展示了在RPN和Fast R-CNN检测网络之间共享卷积层的效果。 为此，我们在4步训练过程的第二步之后停止，使用单独的网络将结果略降至58.7％（RPN + ZF，未共享，表2）。 我们观察到这是因为在第三步中，使用检测器调整的功能来微调RPN时，建议质量得到了改善。

接下来，我们将解开RPN对训练Fast R-CNN检测网络的影响。 为此，我们使用2000 SS提案和ZF网络训练了快速R-CNN模型。 我们修复此检测器，并通过更改测试时使用的建议区域来评估检测mAP。 在这些消融实验中，RPN不与检测器共享特征。

在测试时用300个RPN提案替换SS导致的mAP为56.8％。  mAP的损失是由于培训/测试建议之间的不一致。 该结果用作以下比较的基准。

。。。。其余的都是一些实验，仔细研究再看。

## 5 CONCLUSION 

We have presented RPNs for efficient and accurate region proposal generation. By sharing convolutional features with the down-stream detection network, the region proposal step is nearly cost-free. Our method enables a unified, deep-learning-based object detection system to run at near real-time frame rates. The learned RPN also improves region proposal quality and thus the overall object detection accuracy.

我们已经提出了RPN，以高效，准确地生成 region proposal。 通过与下游检测网络共享卷积特征，区域提议步骤几乎是免费的。 我们的方法使基于深度学习的统一对象检测系统能够以接近实时的帧速率运行。 所学习的RPN还提高了区域提议质量，从而提高了总体目标检测精度。

