# SWINBERT: End-to-End Transformers with Sparse Attention for Video Captioning (Microsoft)
[paper with code](https://paperswithcode.com/paper/swinbert-end-to-end-transformers-with-sparse)  

<img src="./Overview.jpg" alt="drawing" width="750"/> 

## Q1. 文章探究的问题？
### A1.1 Abstract
解决Video Caption任务(给定视频描述其内容)的端到端模型  
***相比于之前的方法中通过额外的2D/3D feature extractor(在图像/视频理解任务上预训练, 例如图像分类或者动作视频)提取视频特征然后生成caption, SwinBERT是端到端的***  
<img src="./Comparison%20between%20previous%20works%20and%20SWINBERT.jpg" alt="drawing" width="350"/>

### A1.2 motion
1. 这些离线训练的feature extractor和下游Video-caption任务在data domain和task formulation上存在明显的gap
2. 如果在密集的视频帧上使用多个feature extractor进行端到端训练, 对于算力的要求很高, 需要设计一种稀疏注意力计算方式

## Q2. 文章主要思路？
### 2.1 Model Architecture
#### 2.1.1 Video Swin Transformer(VidSwin)
```
长时间建模有助于提高视频理解的能力, 但是堆叠大量的帧来捕捉长距离信息对算力的要求很高。因此作者使用VidSwin(Kinetics预训练)作为视频编码器(毕竟也是微软出的), 实现速度和精度的权衡
```
```
给定大小为(T, H, W, 3)的原始视频帧, VidSwin输出(T/2, H/32, W/32, 8C)的特征图, 其中C是通道大小。将特征图展平后输入到Multimodal Transformer Encoder
```

<img src="./3D%20shifted%20window%20of%20VideoSwin.jpg" alt="drawing" height="250"/> <img src="./architecture%20of%20Video%20Swin.jpg" alt="drawing" height="250"/>   

#### 2.1.2 Multimodal Transformer Encoder
结构上就是一个标准的Transformer Encoder, 主要点在:
1. 接收文本和视频两个模态的输入: tokenized caption description和video tokens;
2. seq2seq的方式生成输出;
3. 因果自注意力掩码(causal self-attention mask), 即video tokens之间计算注意力, text token和video tokens以及之前的text token计算注意力

### 2.2 Sparse Attention Mask
理论上, 视频帧数越多, 模型理解视频的能力越好, 但是:  
  1. Multimodal Transformer Encoder在计算attention的消耗就越大  
  2. 由于视频帧间信息冗余的特点, 反而可能会降低性能  

因此作者提出了可学习的稀疏注意力掩码(Sparse Attention Mask), 并用sigmoid激活(值被限制在0~1, 目的在于对attn-weight加权, 提高更有用的连接, 降低无意义连接) :
  1. 在训练阶段, 正则化项会使Attention Mask的右下角部分(对应video token)稀疏化, 降低无意义连接;
  2. 在推理阶段, 可以通过简单地使用0.5的阈值转化为二进制掩码, 进而mask掉部分video token(会降低精度)


### 2.3 整体流程
* train: 
  * 端到端训练
  * Masked Language Modeling作为预训练任务
  * 稀疏注意力掩码作为正则化项
* inference:
  * 视频序列作为唯一输入
  * 以自回归的方式生成输出语句: 模型一次生成一个word token, 将之前生成的word tokens用作Multimodal Transformer Encoder的输入, 重复执行生成, 直到模型输出预定义的结束令牌[EOS]或达到最大输出长度;

<img src="./Overview.jpg" alt="drawing" width="750"/> 

## Q3. 实现和结果
### 3.1 实现细节
- 模型使用Pytorch、Huggingface transforme和DeepSpeed库实现
- VidSwin使用Kinetics-600预训练权重初始化, Multimodal Transformer Encoder随机初始化(这个是有人做过实验说明随机初始化更好的);
- 使用可学习MLP确保Video tokens与word tokens的维度大小相同
- AdamW优化器，并在早期10%的训练步数中使用学习率预热，然后进行线性衰减。
- 更多实现细节可以看文章的附录E

### 3.2 Comparison with state-of-the-art methods
<img src="./Comparison%20with%20state-of-the-art%20methods.jpg" alt="drawing" width="700"/> 

### 3.3 Ablation Study
* 帧数, 稀疏注意力掩码, 可学习的注意力掩码  
  <img src="./ablation%20study1.jpg" alt="drawing" width="700"/> 

* 稀疏注意力掩码对长视频序列的效果: 二进制掩码会使性能下降, 但和Full Attention基线相当或更好  
  <img src="./Effectiveness%20of%20soft%20or%20binary%20sparse%20attention%20mask%20on%20longer%20video%20squence.jpg" alt="drawing" width="350"/> 

* 稀疏注意力掩码的泛化性能: 
  * 跨帧率: 先以低帧速率训练, 然后转到高帧率进一步训练。沿着时间维度的线性插值来扩展所学习的稀疏注意力掩码;
  * 跨数据集：先在一个数据集上训练, 然后在另一个数据集上微调。
  * 实验在两种设置下进行: (1) 迁移整个模型权重; (2)仅迁移稀疏注意力掩码
  
  <img src="./Transferability.jpg" alt="drawing" width="700"/> 

* 不同epoch下attention mask的稀疏性和模型性能  
  <img src="./Training%20behavior%20of%20SWINBERT.jpg" alt="drawing" width="400"/> 

* sparse attn mask的可视化  
  <img src="./visualization%20of%20attn%20mask.jpg" alt="drawing" width="400"/> 

## Q4. 补充
[GIT A Generative Image-to-text Transformer for Vision and Language](https://paperswithcode.com/paper/git-a-generative-image-to-text-transformer)同样是微软在2022年放出的文章, 结构和SwinBERT类似, 不同点在于:
* GIT是Image Encoder, 来源于微软之前的Florence模型, 称为CoSwin(结合CvT和Swin的方法, 将Swin中的patch embedding和patch merging替换为卷积, CvT和Swin也都是微软的)
* GIT没有稀疏化注意力掩码；
* GIT使用Language modeling进行训练(即回归文本), 因为LM训练得更快(作者分析说MLM一次只能训练一部分token的生成，而LM能训练所有token)
* 多模态的Transformer只有6层, 是SwinBERT的一半  

在没有对视频做特别的优化的情况下, 在各个任务上都取得了不错的效果:  
<img src="./Comparison%20with%20state-of-the-art%20methods%20GIT.jpg" alt="drawing" width="400"/> 

<img src="./model%20of%20GIT.jpg" alt="drawing" width="400"/> 