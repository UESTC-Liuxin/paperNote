# mmsegmentation学习笔记

## 前言

mmsegmentation是open-mmlab继mmdetection之后开源的又一大作，专用于语义分割，github直通车：https://github.com/open-mmlab/mmsegmentation。由于喜欢mmd的效果，并忠于open-mmlab良好的编程风格，写下这篇笔记，希望能带给自己收获的同时，能够便利他人。

## mmcv文件夹

在最新的版本中，open-mmlab将mmcv独立了出来，作为一个python的库，为open-mmlab所有的

```
mmcv
├── CONTRIBUTING.md
├── Dockerfile
├── Jenkinsfile
├── LICENSE
├── MANIFEST.in
├── README.md
├── build
│   ├── lib.linux-x86_64-3.7
│   └── temp.linux-x86_64-3.7
├── docs
│   ├── Makefile
│   ├── ...
├── examples
│   ├── config_cifar10.py
│   ├── dist_train_cifar10.sh
│   ├── resnet_cifar.py
│   └── train_cifar10.py
├── mmcv
│   ├── __init__.py
│   ├── __pycache__
│   ├── _ext.cpython-37m-x86_64-linux-gnu.so
│   ├── _flow_warp_ext.cpython-37m-x86_64-linux-gnu.so
│   ├── arraymisc
│   ├── cnn
│   ├── fileio
│   ├── image
│   ├── model_zoo
│   ├── onnx
│   ├── ops
│   ├── parallel
│   ├── runner
│   ├── utils
│   ├── version.py
│   ├── video
│   └── visualization
├── mmcv_full.egg-info
├── requirements.txt
├── setup.cfg
├── setup.py
└── tests
    ├── data
    ├── test_arraymisc.py
    ├── test_cnn
    ├── test_config.py
    ├── test_fileclient.py
    ├── test_fileio.py
    ├── test_fp16.py
    ├── test_image
    ├── test_load_model_zoo.py
    ├── test_logging.py
    ├── test_misc.py
    ├── test_ops
    ├── test_optimizer.py
    ├── test_parallel.py
    ├── test_path.py
    ├── test_progressbar.py
    ├── test_registry.py
    ├── test_runner
    ├── test_timer.py
    ├── test_video
    └── test_visualization.py
```

### open-mmlab总体介绍

这是open-mmlab做的一个python库，可以直接进行安装，是用于计算机视觉研究的基础python库，实际上这里的文件都是针对open-mmlab的众多框架的底层框架库，除了一些底层io，cnn，迭代器等等以外，**open-mmlab都遵从了相同的程序的pipeline**：

=================================================================================================================================================

这不得不提到对于一个python的执行顺序：

<img src="https://img-blog.csdn.net/20180503131738794?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1bnBlbmd0aW5ndGluZw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" style="zoom: 80%;" />

mmlab的程序就是基于此，并配合装饰器python在开始执行的时候扫描整个工程，针对每一个类型的模块（这里指的是数据集，backbone，Loss等）都利用Registry对象储存起来，形成一个字符串作为键，class对象（注意是class本身对象，不是class的实例化对象）作为键值的字典，下面举个例子：

在registry.py里我构建了一个简化版的Registry类，这个类的核心就是register_module方法，这是一个装饰器，用于获取class对象（注意，这里的对象指的是class本身，而不是class的实例，python的所有元素都是对象）。

```python
class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class, module_name=None, force=False):
        self._module_dict[module_name]=module_class


    def register_module(self, name=None, force=False, module=None):
        # use it as a decorator: @x.register_module()
        def _register(*args):
            self._register_module(
                module_class=args, module_name=name, force=force)
            return args

        return _register
```

同时，我在main下面构建了两个类，TestA和TestB。

```PYTHON
from registry import Registry

TEST=Registry('test')
print(TEST)

@TEST.register_module('TestA')
class TestA():

    def __init__(self):
        print('testA.....')
    @classmethod
    def get_class_name(cls):
        return cls.__name__

@TEST.register_module('TestB')
class TestB():

    def __init__(self):
        print('testB.....')
    @classmethod
    def get_class_name(cls):
        return cls.__name__

print(TEST)
```

main文件的执行顺序是：1. 导入registry文件，扫描整个文件。2. 执行语句 3. 最后扫描到TestA，发现装饰器，跳入装饰器中，装饰器中存在一个解析器_register_module函数，此函数的首参数是TestA class本身（注意，这里未实例化），最后把此class 作为对象存在Registry的实例对象TEST的_module_dict中。再接着向下执行TestB。我们可以看看两次的TEST输出（如果没有指定类名TestA），mmlab的做法是解析出类名放在items的key中的：

```python
Registry(name=test, items={})
Registry(name=test, items={'TestA': (<class '__main__.TestA'>,), 'TestB': (<class '__main__.TestB'>,)})
```

mmlab利用这样的方式，构建对应的Resistry->读取config->按config指定的配置，从Resistry读取对应的class，然后实例化后返回。

整个过程就是从config各个字符串的配置，变成可执行的对象。

=========================================================================================================================================================================================

### mmcv/mmcv文件夹

#### runner文件夹

官方对runner模块的介绍是：Runner模块旨在帮助用户以更少的代码开始培训，同时保持灵活性和可配置性。实际上正如文档介绍的一致，runner确实为open-mmlab提供了良好的编程接口，但是mmcv这个库也是建立在open-mmlab这个框架上的，要为我们自己的程序用上，还是不太现实。

```
runner
├── __init__.py
├── __pycache__
├── base_runner.py
├── iter_based_runner.py
├── log_buffer.py
├── checkpoint.py
├── dist_utils.py
├── epoch_based_runner.py
├── fp16_utils.py
├── hooks
│   ├── __init__.py
│   ├── __pycache__
│   ├── checkpoint.py
│   ├── closure.py
│   ├── ema.py
│   ├── hook.py
│   ├── iter_timer.py
│   ├── logger
│   ├── lr_updater.py
│   ├── memory.py
│   ├── momentum_updater.py
│   ├── optimizer.py
│   ├── sampler_seed.py
│   └── sync_buffer.py
├── optimizer
│   ├── __init__.py
│   ├── __pycache__
│   ├── builder.py
│   └── default_constructor.py
├── priority.py
└── utils.py

```

**[BaseRunner](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/base_runner.py)**

这是一个抽象基类，具体用法[抽象基类](https://blog.csdn.net/LaoYuanPython/article/details/92840491)

```python
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``
"""
def __init__(self,
                            model,  #nn.Module
                            batch_processor=None,#用于处理databatch的callable方法，函数原型应为`batch_processor(model, data, train_mode) -> dict`
                            optimizer=None, #优化器，字典或者optimizer对象
                            work_dir=None,
                            logger=None,
                            meta=None):#运行环境信息
```

由于是作为抽象基类，所以有很多抽象方法，需要子类实现。不需要覆写的方法当中，有几个钩子函数的注册函数，我来简单解释一下。

```python
def register_hook(self, hook, priority='NORMAL'):
    pass

def register_hook_from_cfg(self, hook_cfg):
    pass
def call_hook(self, fn_name):
    for hook in self._hooks:
        getattr(hook, fn_name)(self)
def register_momentum_hook(self, momentum_config):
    pass
def register_optimizer_hook(self, optimizer_config):
    pass
def register_checkpoint_hook(self, checkpoint_config):
    pass
def register_logger_hooks(self, log_config):
    pass
def register_training_hooks(self,
                            lr_config,
                            optimizer_config=None,
                            checkpoint_config=None,
                            log_config=None,
                            momentum_config=None):
    pass
```

`register_hook_from_cfg`函数用于将cfg编译为一个个继承于[Hook](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/hook.py)的对象（其中Hook类声明了训练的各个阶段需要调用的函数：`before_run`，`after_run`，`before_train_epoch`，`after_train_epoch`等等，同时，派生类中包含了`lr`,`memeory`,`momentum`等等设置策略。）

`register_hook`函数按照`priority`的级别按顺序插入到`base_runner._hooks = []`中。

`call_hook`根据钩子函数列表依次调用钩子函数。

**[IterBasedRunner](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py)**

这是派生于BasedRunner的类。提供了训练，验证的总流程实现，并定义了一个run函数，在run函数根据mode，实现了训练或者验证的迭代训练过程。

```python
    def run(self, data_loaders, workflow, max_iters, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training     DataLoader的列表
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the                  工作流程元组的列表，比如[('train', 10000),  ('val', 1000)]
                running order and iterations. E.g, [('train', 10000), 
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
            max_iters (int): Total training iterations.                                                      最高迭代次数
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)                                                  #验证输入

        self._max_iters = max_iters
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d iters', workflow, max_iters)
        self.call_hook('before_run')                                                                             #调用函数名带有'before_run'子串的钩子函数

        iter_loaders = [IterLoader(x) for x in data_loaders]                               #将data_loaders的列表转成IterLoader的列表，但其实我没弄明白，这个类是拿来干嘛的
        self.call_hook('before_epoch')
        while self.iter < max_iters:                                                                                #开始迭代
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow                                                                                         #单卡iters就是1，双卡就是2
                if not isinstance(mode, str) or not hasattr(self, mode):    
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)                                                           #搜索本实例中包含mode里'train'/'val'/'test'字符的方法或属性，就找到self.train               
                for _ in range(iters):                                                                                    #
                    if mode == 'train' and self.iter >= max_iters:                                #
                        break
                    iter_runner(iter_loaders[i], **kwargs)                                           #进行一次迭代

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')
```

这个程序是真的很喜欢用`getattr`函数。。。。。。



## mmsegmentation文件总体结构

```
mmsegmentation
├── LICENSE
├── README.md
├── configs   //配置文件，针对每个网络的一写用户参数设置，比如数据集选用，transform操作配置清单，网络结构的配置清单等等。(只是配置清单，没有实现代码)
├── data        //数据集
├── demo    
├── docker
├── docs
├── mmseg  //分割的核心文件，包括数据集class模板，网络的结构的class模板（包括backbone，编码器，解码器），损失函数，训练，测试的api实现等等
├── mmsegmentation.egg-info
├── pytest.ini
├── requirements
├── requirements.txt
├── resources
├── setup.cfg
├── setup.py
├── tests  //测试工具，用于单独测试各个模块是否正常工作
├── tools  //用户工具箱
```

## configs文件夹

```
configs
├── _base_  //这里存的是更细节的配置信息：包括数据集配置，模型的参数配置，学习率等等
├── ann        //这里存的都是些完整的高定制化配置，用什么数据集，用什么网络，用什么学习率策略，是_base_的组装
├── ccnet
├── danet
├── deeplabv3
├── deeplabv3plus
├── encnet
├── fcn
├── fp16
├── gcnet
├── hrnet
├── nonlocal_net
├── ocrnet
├── psanet
├── pspnet
└── upernet
```

示例：/media/Program/CV/Project/SKMT/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x512_20k_voc12aug.py

```python
 _base_ = [
     '../_base_/models/pspnet_r50-d8.py',
     '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
     '../_base_/schedules/schedule_20k.py'
 ]
 model = dict(
     decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))
```

其中,重点在于\__base__文件夹：

```python
_base_
├── datasets      //数据集的配置，包括路径啊，类别啊，预处理的pipeline等等
│   ├── US.py
│   ├── ade20k.py
│   ├── cityscapes.py
│   ├── cityscapes_769x769.py
│   ├── custom_skmt.py
│   ├── pascal_voc12.py
│   └── pascal_voc12_aug.py
├── default_runtime.py
├── file_tree.md
├── models      //配置选用编码器，解码器，loss_decode等等
│   ├── ann_r50-d8.py
│   ├──。。。。。  //省略一下
└── schedules
    ├── schedule_160k.py
    ├── schedule_20k.py
    ├── schedule_40k.py
    └── schedule_80k.py
```



##　ｍｍseg文件夹

==这是最重要的文件夹，里面包含了数据集，网络结构，训练，测试的具体实现== 

```
mmseg
├── VERSION
├── __init__.py
├── apis
│   ├── __init__.py
│   ├── __pycache__
│   ├── inference.py
│   ├── test.py
│   └── train.py
├── core
│   ├── __init__.py
│   ├── __pycache__
│   ├── evaluation
│   ├── seg
│   └── utils
├── datasets                                      //用于自定义数据集，以及数据集构建器
│   ├── __init__.py
│   ├── __pycache__
│   ├── ade.py
│   ├── builder.py                         //每个dataset文件夹下和model文件夹下都有一个builder文件，作用是利用cfg的字符串信息构建对应的实例对象，比如dataset对象等。
│   ├── cityscapes.py
│   ├── custom.py
│   ├── dataset_wrappers.py
│   ├── pipelines
│   ├── skmt.py
│   ├── us.py
│   └── voc.py
├── file_tree.md
├── models                                     
│   ├── __init__.py
│   ├── __pycache__
│   ├── backbones                   //存放了各种各样的backbone的实现
│   ├── builder.py                    //此builder同上
│   ├── decode_heads             
│   ├── losses
│   ├── segmentors              //分割器，目前是CascadeEncoderDecoder分割器，这是一个nn.Module的派生类
│   └── utils
├── ops
│   ├── __init__.py
│   ├── __pycache__
│   ├── encoding.py
│   ├── separable_conv_module.py
│   └── wrappers.py
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   ├── collect_env.py
│   └── logger.py            //logging的二次封装
└── version.py
```

### [builder.py](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/builder.py)

在`mmseg/seg/`,`mmseg/datasets`,`mmseg/model`里都有builder函数，用于将config信息 "编译" 实例对象，比如module模块。

下面针对于`mmseg/model`下的builder进行说明，期间会给出具体的实例。

```python
from mmcv.utils import Registry, build_from_cfg
from torch import nn

SEGMENTORS = Registry('segmentor')

def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    return build(cfg, SEGMENTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
```

- 首先调用build_segmentor，将train(ConfigDict)和test(ConfigDict)打包成字典传入build函数中。SEGMENTORS是模块注册器，利用装饰器对所有的SEGMENTORS模型进行注册，包括了：`CascadeEncoderDecoder`类和`EncoderDecoder`类。

- 最后在build中判断当前的cfg对象是list还是ConfigDict对象（默认是ConfigDict对象）。对于ConfigDict对象，需要对其进行“编译”，提取type关键字。
- 利用registry.get()获取class，并进行实例化。（这里获取的是`EncoderDecoder`）。并将args信息，也就是之前提到train和test的cfg信息。

### models/segmentators

```
segmentators                                                                                             
├── base.py                                                                                          
├── cascade_encoder_decoder.py                                                                           
├── encoder_decoder.py                                                                                
├── __init__.py
```

这里目前只介绍最重要的部分segmentators。什么

此文件夹下包含了三个文件，这三个文件分别包含了`class BaseSegmentor(nn.Module)`，`EncoderDecoder(BaseSegmentor)`，`CascadeEncoderDecoder(EncoderDecoder)`这三个类，可以看到是依次继承的关系。

[**BaseSegmentor**](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/segmentors/base.py)

此类中定义了一些基本的属性，以及抽象方法（父类中用`@abstractmethod`声明为抽象方法，规定了子类必须实现才能实例化）。

这里只介绍一些没有被重写的方法:

```python
#这里只给出方法原型，不给出具体实现，具体实现看源码
@auto_fp16(apply_to=('img', ))  #这是一个转fp16的装饰器，对'img'进行了一个转换。
def forward(self, img, img_metas, return_loss=True, **kwargs):
    """Calls either :func:`forward_train` or :func:`forward_test` depending   根据是否return loss选择是利用哪个前向传播
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
    if return_loss:
        return self.forward_train(img, img_metas, **kwargs)
    else:
        return self.forward_test(img, img_metas, **kwargs)
```

说到这里，不得不一句pytorch对Module对象可以直接在实例化对象后，直接利用实例化对象作为函数的形式，调用forward函数的原理。[python `__call__`](https://blog.csdn.net/qq_20549061/article/details/107891343)

```python
#静态方法，对loss进行解析，输入的是一个字典，每一个元素都是损失或者acc，程序中提取各个模块的子损失，等比加起来成为总损失，并将损失信息存入log里。
#实际例子：losses={'decode.loss_seg': tensor(1.6582, device='cuda:0', grad_fn=<MulBackward0>), 'decode.acc_seg': tensor([42.7540], device='cuda:0'), 'aux.loss_seg': #tensor(0.8023, device='cuda:0', grad_fn=<MulBackward0>), 'aux.acc_seg': tensor([42.0382], device='cuda:0')}
@staticmethod
def _parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

            Args:
                losses (dict): Raw output of the network, which usually contain
                    losses and other necessary information.

            Returns:
                tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                    which may be a weighted sum of all losses, log_vars contains
                    all the variables to be sent to the logger.
            """
    log_vars = OrderedDict()  #通常字典都是无序的hashmap，但这个是有序的
    for loss_name, loss_value in losses.items(): 
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
         else:
             raise TypeError(
                 f'{loss_name} is not a tensor or list of tensors')

     loss = sum(_value for _key, _value in log_vars.items()#将所有的损失加起来
                               if 'loss' in _key)

     log_vars['loss'] = loss
     for loss_name, loss_value in log_vars.items():
          # reduce loss when distributed training
          if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
           log_vars[loss_name] = loss_value.item()
                   
     return loss, log_vars
```

[EncoderDecoder](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/segmentors/encoder_decoder.py)

这是默认使用的网络结构的class，继承于BaseSegmentor。

查看代码中的描述：EncoderDecoder通常由backbone，decode_head,auxiliary_head组成。auxiliary_head只用于训练时deep supervision。

我们查看实例化参数：

```python
def __init__(self,
                            backbone,
                            decode_head,
                            neck=None,
                            auxiliary_head=None,
                            train_cfg=None,
                            test_cfg=None,
                            pretrained=None):
      pass
```

在构造函数`__init__`中，首先根据config信息，利用builder转换为了实际的模型，并调用`init_weights`函数初始化权重或者加载预训练权重，加载预训练权重的操作在mmcv/mmcv/runner/checkpoint.py中。

然后看最关键的代码，前向传播，代码的前向传播是由

## tools文件夹

```
├── benchmark.py
├── convert_datasets
│   ├── cityscapes.py
│   └── voc_aug.py
├── dist_test.sh
├── dist_train.sh     //分布式训练（这里指的是单机多卡）
├── file_tree.md    
├── get_flops.py   
├── print_config.py
├── publish_model.py
├── slurm_test.sh
├── slurm_train.sh  //多机多卡
├── test.py
└── train.py
```

### train.py

由于这个文件基本都是调用的其余文件的类和方法，因此，这里做一个流程介绍，会涉及到其余文件。

**cfg对象**

cfg是一个Config的对象。这是官方解释：将各种文本设置变成可访问的属性。

```python
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """
```

**参数解析**

首先看参数代码：

```bash
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
```

只有一个位置参数 `config`，configs文件的绝对/相对路径，通常需要把工作目录设置在mmsegmentation的根目录；

可选参数，`work_dir`,默认会建立在根目录

可选参数，`resume-from`， 选择用于恢复的模型，以避免从头训练。

可选参数，`no-validate`，action设置为store_true（命令若包含此参数，就不对模型进行训练评估，默认是要进行评估）。
可选参数，`gpus`，非分布式训练下单机多卡的gpu数量（>0）。

可选参数，`gpu-ids`，gpu的id号 ，nargs =‘+’。命令格式为`--gou-ids 0 1 2`。

可选参数，`seed`，随机数种子数，int型。

可选参数，`deterministic` ，是否取用cudnn加速，默认没有。

可选参数，`options`，action为自定义操作。

可选参数，`launcher`，没懂，反正我选择pytorch也报了错，默认就不要选了。

可选参数，`local_rank`，多机分布式训练的参数，不用管（贫穷让我无法了解这些高端操作）

**torch.backends.cudnn.benchmark**

https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317

**model = build_segmentor( cfg.model, ...)**

开始构建分割模型，开始mmsegmentation的高级装逼代码之路。（讲道理，这个代码对我来说，确实有些复杂）。


## 基本网络结构

#### backbone

```python
ResNetV1c(
  (stem): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
    (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace=True)
  )
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): ResLayer(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer2): ResLayer(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer3): ResLayer(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer4): ResLayer(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
)
```

psphead

```python
PSPHead(
  input_transform=None, ignore_index=255, align_corners=False
  (loss_decode): CrossEntropyLoss()
  (conv_seg): Conv2d(512, 17, kernel_size=(1, 1), stride=(1, 1))
  (dropout): Dropout2d(p=0.1, inplace=False)
  (psp_modules): PPM(
    (0): Sequential(
      (0): AdaptiveAvgPool2d(output_size=1)
      (1): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
    (1): Sequential(
      (0): AdaptiveAvgPool2d(output_size=2)
      (1): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
    (2): Sequential(
      (0): AdaptiveAvgPool2d(output_size=3)
      (1): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
    (3): Sequential(
      (0): AdaptiveAvgPool2d(output_size=6)
      (1): ConvModule(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activate): ReLU(inplace=True)
      )
    )
  )
  (bottleneck): ConvModule(
    (conv): Conv2d(4096, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activate): ReLU(inplace=True)
  )
)
```

auxliry-head

```python
FCNHead(
  input_transform=None, ignore_index=255, align_corners=False
  (loss_decode): CrossEntropyLoss()
  (conv_seg): Conv2d(256, 17, kernel_size=(1, 1), stride=(1, 1))
  (dropout): Dropout2d(p=0.1, inplace=False)
  (convs): Sequential(
    (0): ConvModule(
      (conv): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activate): ReLU(inplace=True)
    )
  )
)
```

