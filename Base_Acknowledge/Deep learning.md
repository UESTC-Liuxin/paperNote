# Deep learning

## 作者：

Yann LeCun1,2, Yoshua Bengio3 & Geoffrey Hinton4,5  

## 翻译

Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction. These methods have dramatically improved the state-of-the-art in speech recognition, visual object recognition, object detection and many other domains such as drug discovery and genomics. Deep learning discovers intricate structure in large data sets by using the backpropagation algorithm to indicate how a machine should change its internal parameters that are used to compute the representation in each layer from the representation in the previous layer. Deep convolutional nets have brought about breakthroughs in processing images, video, speech and audio, whereas recurrent nets have shone light on sequential data such as text and speech.

深度学习允许由多个处理层组成的计算模型学习具有多个抽象级别的数据表示。 这些方法极大地改善了语音识别，视觉对象识别，对象检测以及许多其他领域（例如药物发现和基因组学）的最新技术。 深度学习通过使用反向传播算法指示机器应如何更改其内部参数来发现大型数据集中的复杂结构，这些内部参数用于根据上一层的表示来计算每一层的表示。 深层卷积网络在处理图像，视频，语音和音频方面带来了突破，而递归网络则对诸如文本和语音之类的顺序数据有所启发。

Machine-learning technology powers many aspects of modern society: from web searches to content filtering on social net works to recommendations on e-commerce websites, and it is increasingly present in consumer products such as cameras and smartphones. Machine-learning systems are used to identify objects in images, transcribe speech into text, match news items, posts or products with users’ interests, and select relevant results of search.Increasingly, these applications make use of a class of techniques called deep learning.

机器学习技术为现代社会的各个方面提供了强大的动力：从网络搜索到社交网络上的内容过滤，再到电子商务网站上的推荐，机器学习技术越来越多地出现在诸如相机和智能手机之类的消费产品中。 机器学习系统用于识别图像中的对象，将语音转录为文本，将新闻项，帖子或产品与用户的兴趣进行匹配以及选择相关的搜索结果。这些应用程序越来越多地使用一类称为深度学习的技术 。

```
make use of : 利用
```

Conventional machine-learning techniques were limited in their ability to process natural data in their raw form. For decades, constructing a pattern-recognition or machine-learning system required careful engineering and considerable domain expertise to design a feature extractor that transformed the raw data (such as the pixel values of an image) into a suitable internal representation or feature vector from which the learning subsystem, often a classifier, could detect or classify patterns in the input.

常规的机器学习技术在处理原始格式的自然数据的能力方面受到限制。 几十年来，构建模式识别或机器学习系统需要认真的工程设计和相当多的领域专业知识，才能设计特征提取器，以将原始数据（例如图像的像素值）转换为合适的内部表示或特征向量， 学习子系统（通常是分类器）可以检测或分类输入中的模式。

Representation learning is a set of methods that allows a machine to be fed with raw data and to automatically discover the representations needed for detection or classification. Deep-learning methods are representation-learning methods with multiple levels of representation, obtained by composing simple but non-linear modules that each transform the representation at one level (starting with the raw input) into a representation at a higher, slightly more abstract level. With the composition of enough such transformations, very complex functions can be learned. For classification tasks, higher layers of representation amplify aspects of the input that are important for discrimination and suppress irrelevant variations. An image, for example, comes in the form of an array of pixel values, and the learned features in the first layer of representation typically represent the presence or absence of edges at particular orientations and locations in the image. The second layer typically detects motifs by spotting particular arrangements of edges, regardless of small variations in the edge positions. The third layer may assemble motifs into larger combinations that correspond to parts of familiar objects, and subsequent layers would detect objects as combinations of these parts. The key aspect of deep learning is that these layers of features are not designed by human engineers: they are learned from data using a general-purpose learning procedure.

表示学习是一组方法，这些方法允许向机器提供原始数据并自动发现检测或分类所需的表示。 深度学习方法是具有表示形式的多层次的表示学习方法，它是通过组合简单但非线性的模块而获得的，每个模块将一个级别（从原始输入开始）的表示转换为更高，稍微抽象的级别的表示 。 有了足够多的此类转换，就可以学习非常复杂的功能。 对于分类任务，较高的表示层会放大输入中对区分非常重要的方面，并抑制不相关的变化。 ==例如，图像以像素值阵列的形式出现，并且在表示的第一层中学习的特征通常表示图像中特定方向和位置上边缘的存在或不存在。 第二层通常通过发现边缘的特定布置来检测图案，而与边缘位置的微小变化无关。 第三层可以将图案组装成与熟悉的对象的各个部分相对应的较大组合，并且随后的层将把对象检测为这些部分的组合。 深度学习的关键方面是这些层的功能不是由人类工程师设计的：它们是使用通用学习过程从数据中学习的。==

Deep learning is making major advances in solving problems that have resisted the best attempts of the artificial intelligence community for many years. It has turned out to be very good at discovering intricate structures in high-dimensional data and is therefore applicable to many domains of science, business and government. In addition to beating records in image recognition1–4 and speech recognition5–7, it has beaten other machine-learning techniques at predicting the activity of potential drug molecules8, analysing particle accelerator data9,10, reconstructing brain circuits11, and predicting the effects of mutations in non-coding DNA on gene expression and disease12,13. Perhaps more surprisingly, deep learning has produced extremely promising results for various tasks in natural language understanding14, particularly topic classification, sentiment analysis, question answering15 and language translation16,17.

==深度学习在其领域的应用：cience, business and government；image/speech recognition；activity of potential drug molecules；analysing particle accelerator data9,10, reconstructing brain circuits11, and predicting the effects of mutations in non-coding DNA on gene expression and disease12,13；atural language understanding==。

We think that  deep learning will have many more successes in the near future because it requires very little engineering by hand, so it can ==easily take advantage of increases== in the amount of available computation and data. New learning algorithms and architectures that are currently being developed for deep neural networks will only accelerate this progress.

The most common form of machine learning, deep or not, is supervised learning. Imagine that we want to build a system that can classify images as containing, say, a house, a car, a person or a pet. We first collect a large data set of images of houses, cars, people and pets, each labelled with its category. During training, the machine is shown an image and produces an output in the form of a vector of scores, one for each category. We want the desired category to have the highest score of all categories, but this is unlikely to happen before training.

We compute an objective function that measures the error (or distance) between the output scores and the desired pattern of scores. The machine then modifies its internal adjustable parameters to reduce this error. These adjustable parameters, often called weights, are real numbers that can be seen as ‘knobs’ that define the input–output function of the machine. In a typical deep-learning system, there may be hundreds of millions of these adjustable weights, and hundreds of millions of labelled examples with which to train the machine.

解释了损失函数和权重。

To properly adjust the weight vector, the learning algorithm computes a gradient vector that, for each weight, indicates by what amount the error would increase or decrease if the weight were increased by a tiny amount. The weight vector is then adjusted in the opposite direction to the gradient vector.

解释梯度下降

The objective function, averaged over all the training examples, can be seen as a kind of hilly landscape in the high-dimensional space of weight values. The negative gradient vector indicates the direction of steepest descent in this landscape, taking it closer to a minimum, where the output error is low on average.

梯度下降的目的就是要找到landscape中平均loss的最小值处。（注意：landscape实际上的峰值为average loss ，而其余轴为整个w，b的取值）

In practice, most practitioners use a procedure called stochastic gradient descent (SGD). This consists of showing the input vector for a few examples, computing the outputs and the errors, computing the average gradient for those examples, and adjusting the weights accordingly. The process is repeated for many small sets of examples from the training set until the average of the objective function stops decreasing. It is called stochastic because each small set of examples gives a noisy estimate of the average gradient over all examples. This simple procedure usually finds a good set of weights surprisingly quickly when compared with far more elaborate optimization techniques18. After training, the performance of the system is measured on a different set of examples called a test set. This serves to test the generalization ability of the machine — its ability to produce sensible answers on new inputs that it has never seen during training.

介绍SGD,SGD就是对整个样本的一个小批次采样，以小批次样本的损失和平均梯度对整个样本集进行无偏差估计，最后进行梯度更新。

Many of the current practical applications of machine learning use linear classifiers on top of hand-engineered features. A two-class linear classifier computes a weighted sum of the feature vector components.If the weighted sum is above a threshold, the input is classified as belonging to a particular category.

解释了线性回归的常见做法，对手工设计的特征向量进行加权求和，设置阈值来进行分类。

Since the 1960s we have known that linear classifiers can only carve their input space into very simple regions, namely half-spaces separated by a hyperplane19. But problems such as image and speech recognition require the input–output function to be insensitive to irrelevant variations of the input, such as variations in position, orientation or illumination of an object, or variations in the pitch or accent of speech, while being very sensitive to particular minute variations (for example, the difference between a white wolf and a breed of wolf-like white dog called a Samoyed). At the pixel level, images of two Samoyeds in different poses and in different environments may be very different from each other, whereas two images of a Samoyed and a wolf in the same position and on similar backgrounds may be very similar to each other. 

图像和语音识别要求对输入的不相关的信息不敏感。在像素层面上，统一物体的我