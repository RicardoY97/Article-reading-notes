# Revealing Single Frame Bias for Video-and-Language Learning
[paper with code](https://paperswithcode.com/paper/revealing-single-frame-bias-for-video-and)  

<img src="./SINGULARITY%20model%20overview.jpg" alt="drawing" width="700"/> 

## Q1. 文章探究的问题？
### A1. 多帧视频输入的冗余性, 探索用于视频和语言学习的单帧模型, 通过大规模的预训练和推理时适当的帧集成策略, 不考虑时间信息的单帧训练模型可以比使用多帧进行训练的现有方法获得更好的性能


## Q2. 文章主要思路？
### 2.1 Model Architecture
* vision encoder: 图像视觉编码器, 如ViT
* language encoder: 任意的语言编码器, 如BERT
* multi-modal encoder: transformer encoder, 每层为(self-attn + cross-attn + FFN)

一个视频V=[f1, f2, ...]和对应的文本S:  
* 在训练阶段, 从V中随机采样一帧作为vision encoder的输入, S作为language encoder的输入, 编码后的序列送入multi-modal encoder做交互  
  <img src="./multi-modal%20encoder%20while%20training.jpg" alt="drawing" width="600"/> 

* 在推理阶段, 从V中采样T帧, 每一帧单独作为vision encoder的输入, 将vision encoder的T帧输出concate送入multi-modal encoder
  <img src="./multi-modal%20encoder%20while%20infering.jpg" alt="drawing" width="600"/>

### 2.2 Pre-Training Objectives
1. 视觉-文本对比：一种对比损失, 将来自视觉和语言编码器的视觉特征和文本特征对齐(做了池化)。
2. 掩码语言建模(MLM)：使用多模态编码器从其文本和视觉上下文中预测masked token。
3. 视觉-文本匹配：用多模态编码器预测视觉-文本对的匹配分数。


## Q3. 实现和结果
### 3.1 实现细节
* 数据:  
  * 图像文本对: COCO, Visual Genome, SBU Captions, CC3M, CC12M;
  * 视频文本对: WebVid
  * 将上述数据分成两个不同的数据集: (1) 5M语料库, 包含来自CC3M+WebVid的5.44M图像和视频; (2) 17M语料库, 包含上述所有数据集的17.28万图像和视频。

* vision encoder: 在ImageNet-21K上预先训练的BEiT-BASE
* language encoder: 从BERT-BASE的前9层开始初始化
* multi-modal encoder: 从同一BERT-BASE模型的最后3层初始化的(cross-attn是随机初始化的)
  
* AdamW, lr=1e-4, epoch=10(在第一个epoch预热, 然后在剩下的训练中余弦衰减到1e-6), 混合精度训练, 单卡GPU batchsize=128, 3张A100. img_size=224*224, 在训练过程中随机调整大小、裁剪和翻转。
* 预训练在5M语料库上需要大约1天, 在17M语料库上需要4天。

### 3.2 Text-to-Video Retrieval
```
1. 在FT过程中, 使用与预训练相同的架构(不使用MLM损失)。初始学习率1e-5, 余弦衰减到1e-6, batchsize=32, epochs=5.
2. 训练中, 每个视频只使用一帧。测试时, MSRVTT和DiDeMo的每个视频使用12帧, ActivityNet-Captions使用32帧(它的视频更长)
3. 在单个A100 GPU上, 这种微调对于MSRVTT大约需要1.5小时, 对于ActivityNet-Captions或DiDeMo大约需要0.5小时
```
  <img src="./result%20on%20text-to-video%20retrieval.jpg" alt="drawing" width="600"/> 


### 3.3 Video Question Answering
```
1. Video Question Answering: 给定一个视频(通常伴随一个文本问题), 生成问题的回答或从一组候选中选择最合适的答案。
2. 对于开放式QA任务, 添加一个额外的multi-modal decoder(从预训练的multi-modal encoder), 将multi-modal encoder的输出作为cross-attn的输入, 并以[CLS]作为start token对answer文本进行解码;
3. 初始学习率1e-5, 在前半个epoch预热学习率, 余弦衰减到1e-6, batchsize=32, epochs=10. 在单个A100 GPU上微调, MSRVT-QA大约需要4个小时, ActivityNet QA大约需要1个小时。对每个视频使用一帧进行训练, 使用12帧进行测试。
```
  <img src="./result%20on%20video%20question%20answering.jpg" alt="drawing" width="600"/> 


### 3.4 空间建模和时间建模的探究
* 动机：
  ```
  3.3和3.4的结果揭示了一个有趣的结论, 即当下流行的视频语言数据集具有强烈的静态外观偏差(static appearance biases): 与处理多个时间顺序帧的模型相比, 我们的模型在每个训练步骤中每个视频只使用一帧, 同样实现了有竞争力的性能。这说明当前数据集的评估指标并不能很好地表现出模型是否能够识别相邻视频帧之间的细粒度时间关系。
  ```
* 方案: 提出两个新的数据集来补充现有的数据集, 来进行更全面的评估: 
  ```
  我们从视频动作识别中获得灵感, 并将时间密集的动作识别数据集Something-Something V2转换为视频和语言数据集。SSv2数据集的一个独特特性是, 视频通常需要细粒度的时间建模来正确预测其动作类。基于SSv2视频和标注, 我们定义了两个文本到视频的检索任务:
  1. SSv2-Template Retrieval: 使用SSv2中的174个模板（例如, “向空中扔东西并抓住它”）作为检索视频的文本查询。使用168913个SSv2训练集视频进行训练。由于测试集的ground-truth不可用, 使用验证集作为测试集(为每个模板采样12个视频, 总共2088个视频用于测试)
  2. SSv2-Label Retrieval: 使用SSv2中的标签注释（例如, “向空中扔钥匙并抓住它”）作为文本查询来检索视频。同样, 168913个视频用于训练, 2088个视频用于测试。

  两个任务相比, Template Retrieval的文本查询不涉及具体的object, 需要对action具有更深的理解, Label Retrieval提供了对静态和时间理解的更全面的评估。

  ```
  <img src="./SSv2%20examples.jpg" alt="drawing" width="700"/> 

* 实验:
  * baseline: Frozen和CLIP4Clip(seqTransf版本). Frozen使用space-time transformer进行视频编码, CLIP4Clip是基于CLIP模型的扩展, 具有额外的4层temporal transformer encoder
  * 多帧变体: 在vision encoder之后添加2层temporal transformer encoder, 将其输出作为multi-modal encoder的输入. 从单帧预训练模型初始化, 使用WebVid视频(4帧)进行第二阶段的视频预训练。初始学习率5e-5, 训练5个epochs。

* 结论: 
  * 单帧模型(5M)性能很差, 它表明, 利用静态外观偏差的模型无法解决新任务;
  * 在添加了2层时间编码器之后, 4帧时间模型比单帧模型获得了显著的性能提升, 超过了baseline;
  * 当使用更多的预训练数据(17M)时, SSv2-Label Retrieval有明显的性能增益, 而SSv2-Template Retrieval上性能变化不大。这表明, SSv2-Label Retrieval需要静态和时间建模, 增强两者都将提高任务性能。对于SSv2-Template Retrieval, 由于其文本查询中不存在对象, 因此它主要依赖时间建模;  
<img src="./Comparison%20to%20existing%20methods%20on%20SSv2%20tasks.jpg" alt="drawing" width="600"/> 

### 3.5 多帧信息聚合策略的探究
1. 对于MSRVTT检索和ActivityNet QA, 早期融合策略（concat）比三种后期融合策略（lse、max、mean）都有显著的优势, 表明了在进行预测时考虑整个视频的重要性。
2. 一般来说, 对于所有融合策略, 在推理时使用更多的帧可以提高模型性能。然而, 对于后期融合策略, 有时使用更多的帧会损害性能, 例如, 对于ActivityNet QA, 超过4帧的推理不如4帧的最大池推理。这一结果与ClipBERT中的MSRVT-QA结果一致。我们认为后期融合的低性能和不稳定是因为其视频级预测是通过聚合帧级预测获得的, 而这些帧级预测可能是不准确和不稳定的, 因为它们是使用每帧中的不完整信息单独预测的, 忽略了它们的上下文.  
<img src="./frame%20ensemble%20strategy.jpg" alt="drawing" width="600"/> <img src="./Prediction%20score%20distribution%20for%20a%20MSRVTT-MC%20example.jpg" alt="drawing" width="667"/> 

### 3.6 预训练数据规模的探究
1. 1帧和4帧模型都从大规模的预训练中受益匪浅。
2. 随着预训练数据大小的增加, 1帧和4帧模型之间的性能差距几乎单调减小。这表明, 当在足够数量的数据上进行预训练时, 单帧模型的性能可能非常接近多帧(需要细粒度时间建模的任务例外)。
  ```
  一种可能的解释是, 由于不完整的上下文和随机采样, 单帧训练比多帧训练噪声更大, 单帧预测通常不准确, 并且不如多帧预测稳定, 在这些情况下, 预训练是有帮助的。同时, 单帧训练要求模型从单帧中提取更多信息, 而多帧模型可以依赖于来自多帧的丰富来源。因此, 对于下游任务, 更重要的是单帧模型从强的预训练模型初始化。
  ```
<img src="./Model%20performance%20as%20a%20function%20of%20pre-training%20data%20size.jpg" alt="drawing" width="600"/>

### 3.6 Training Efficiency
<img src="./training%20efficiency.jpg" alt="drawing" width="600"/>