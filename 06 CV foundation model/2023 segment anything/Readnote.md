# Segment Anything(2023, Meta)
[paper with code](https://arxiv.org/abs/2302.03024v1)  
[demo](https://segment-anything.com/)  
ps. 目前不支持text prompt

## Q1. 文章针对的问题？
### A1. a foundation model for image segment task

## Q2. 文章要验证的假设是什么？
### A2. 
(1) foundation models拥有出色的encoder能力和迁移能力, 而且既有的实验表明, 数据量越大、模型scale越大, zero and few-shot越强, 甚至可以和有监督模型相抗衡;  
(2) 通过prompt engineering可以将foundation model在新的数据分布上实现zero-shot generalization;
```
1. What task will enable zero-shot generalization? 
2. What is the corresponding model architecture?
3. What data can power this task and model?
```

## Q3. 有哪些相关研究？如何归类？
### A3. 
(1) foundation models: CLIP、ALIGN、DALL·E; 

## Q4. 文章的解决方案是什么？
### 4.1 Abstract
<img src="./task model and data of SAM.png" alt="drawing" width="800"/>     

### 4.2 Segment Anything Task
```
inspiration from NLP, where the next token prediction task is used for foundation model pre-training and to solve diverse downstream tasks via prompt engineering 
```
1. Task:   
   (1) target: 在给定任意prompt的情况下生成有效的分割掩码, 即使在prompt有歧义或者指向多个目标时至少输出其中一个目标的合理掩码(这类似于让语言模型在有歧义的prompt下输出coherent response);
   (2) prompt: foreground/background points, a rough box or mask, free-form text, 任意指示分割图像目标的信息都可以;     
   (3) reason: 选择此任务是因为它会产生一种自然的预训练算法和一种通过提示将零样本转移到下游分割任务的通用方法;  

   <img src="./ambiguous point prompt.png" alt="drawing" width="300"/> 
2. Pre-training:   
   (1) target:  simulates a sequence of prompts (e.g., points, boxes, masks) for each training sample and compares the model’s mask predictions against the ground truth.  
   (2) performing well at this task is challenging and requires specialized modeling and training loss choices;

3. Zero-shot transfer:  
   预训练任务使得模型在推理时对任何prompt做出适当响应的能力，因此下游任务可以通过设计适当的prompt来解决。

4. Related tasks:  
   (1) Segmentation is a broad field: there’s interactive segmentation, edge detection, super pixelization, object proposal generation, foreground segmentation, semantic segmentation, instance segmentation, panoptic segmentation, etc.  
   (2) The goal of our promptable segmentation task is to produce a broadly capable model that can adapt to many (though not all) existing and new segmentation tasks via prompt engineering.   

5. Discussion:  Prompting and composition
   (1) 单个模型能够以可扩展的方式使用, 并可能完成在模型设计之初未知的任务, 例如CLIP是DALL·E图像生成系统的文本图像对齐组件. 基于prompt engineering的composable system比专门为固定任务训练的系统相比, 具有更广泛的应用;
   (2) 从composition角度来比较可promptable segmentation和interactive segmentation也很有趣：虽然interactive segmentation是在考虑用户的情况下设计的，但promptable segmentation也可以组成一个更大的算法系统;   

### 4.2 Segment Anything Model
<img src="./SAM overview.png" alt="drawing" width="800"/>   

1. Image encoder: an MAE pre-trained ViT minimally adapted to process high resolution inputs;   
2. Prompt encoder: 按照不同的prompt类型分为  
   (1) points and boxes作为positional encodings, summed with learned embeddings for each prompt type;   
   (2) free-form text是用CLIP现成的文本编码器来编码;  
   (3) masks使用卷积嵌入, 并与image embedding逐元素求和;

3. Mask decoder:  inspired by DETR anf MaskFormer  
   (1) modified decoder block: prompt self-attention and cross-attention in two directions (prompt-to-image embedding and vice-versa) to update all embeddings.  
   (2) dynamic mask prediction head: 在2 blocks之后将image embedding上采样, MLP将output token映射到dynamic linear classifier, 然后该分类器计算每个图像位置的掩码前景概率;  

4. Resolving ambiguity:  
   (1) predict multiple output masks for a single prompt: 当输出只有1个且prompt存在时, 模型将会平均输出多个有效的mask, 文中选择对每个prompt输出3个mask(一般情况下, mask最多有3层whole, part, subpart);   
   (2) 在训练过程中, 只对loss最小的mask作反向传播. 为了对mask排序, 模型为每个mask预测一个置信度分数(i.e., estimated IoU);  

5. Efficiency:  
   整体模型设计在很大程度上是出于效率的考虑。给定预编译的image embedding, prompt encoder和mask decoder在网络浏览器中运行,在CPU上，以~50ms为单位。这种性能使我们的模型能够无缝、实时地进行交互式提示。

6. Losses and training:  
   (1) losses: focal loss and dice loss;  
   (2) train for the promptable segmentation task using ***a mixture of geometric prompts***;   
   (3) we simulate an interactive setup by randomly sampling prompts in 11 rounds per mask, allowing SAM to integrate seamlessly into our data engine.  

### 4.3 Segment Anything Data Engine
1. Assisted-manual stage:   
   (1) 使用公开分割数据集训练SAM;  
   (2) 专业注释人员通过基于SA的浏览器交互式分割工具点击前景/背景对象点来标记mask;    
   (3) 在充分的数据注释之后，仅使用新注释的掩码对SAM进行retrain. 随着数据量增大, 图像编码器从ViT-B扩展到ViT-H(其他架构细节同样), 总共对模型进行了6次retrain。随着模型的改进，每个mask的平均标注时间从34秒减少到14秒(比COCO的mask标注快6.5倍, 仅比使用极值点的边界框标记慢2倍)。随着SAM的改进，每张图像的平均mask数量从20个增加到44个;  
   (4) 在这个阶段从120k张图像中收集了430万个mask。  

2. Semi-automatic stage.  
   (1) aim: 增加mask的多样性，提高模型segment anything的能力;  
   (2) method: 先检测出高置信度的mask, 让标注人员在此基础上标注没有标注的mask.  
   ```
   To detect confident masks, we trained a bounding box detector [84] on all first stage masks using a generic “object” category.
   [84]: Faster R-CNN
   ``` 
   (3) 与第一阶段一样，定期根据新收集的数据对模型进行retrain(5次)。每个掩码的平均标注时间回到了34秒(不包括自动掩码)，因为这些对象更难标记。每张图像的平均mask数量从44个增加到72个(包括automatic mask);  
   (4) 在这个阶段从180k张图像中额外收集了590万个mask(合计1020万个mask);    

3. Fully automatic stage.
   (1) feasibility: 首先之前已经收集足够的标注数量来优化模型, 其次模型已经具有了在歧义下预测有效mask的能力;  
   (2) method:  
   * 用32×32的规则网格点作为prompt, 为每个点预测一组可能对应于有效对象的掩码(如果一个点位于part或subpart上, 模型将返回subpart, part, and whole object). 
   * IoU prediction module用于选择高置信度掩码. 
   * 只选择稳定的掩码(如果在0.5−δ和0.5+δ处对概率图进行阈值处理会导致类似的掩码，则认为掩码是稳定的).
   * 在选择了高置信度和稳定的掩码后，我们应用NMS来过滤。
   * 为了进一步提高较小遮罩的质量，我们还处理了多个重叠的放大的图像抠图。
   (3) 将automatic mask generation应用于数据集中的所有1100万张图像，总共生成了11亿个高质量mask。


### 4.4 Segment Anything Dataset
1. Images: 1100万张, 高分辨率(平均3300×4950), 发布最短边设置为1500像素的下采样图像;  
2. Masks: 1.1B masks, 99.1% of which were generated fully automatically.  
3. Mask quality: 
   ```
   为了估计mask质量，随机采样了500张图像（约5万个mask），并要求专业注释人员提高这些图像中所有mask的质量。这一过程产生了一对自动预测和专业校正的口罩。我们计算了每对之间的IoU，发现94%的pair的IoU大于90%（97%的pair的IoU大于75%）。为了进行比较，先前的工作的一致性为85-91%IoU。我们在§7中的实验通过人类评级证实，相对于各种数据集，mask质量很高，并且在automatic mask上训练我们的模型几乎与使用数据引擎产生的所有mask一样好。
   ```
4. Mask properties
   
### 4.5 Segment Anything RAI Analysis
1. Geographic and income representation.  
2. Fairness in segmenting people.  

## Q5. 文章的实验是怎么设计的？
### A5. Zero-Shot Transfer Experiments
**Implementation**:   
* an MAE pre-trained ViT-H image encoder;  
* trained on SA-1B (only automatically generated masks);

### 5.1 Zero-Shot Single Point Valid Mask Evaluation
* Results:  
    <img src="./samples of sero-shot transfer capabilities.png" alt="drawing" width="1000"/> 
    <img src="./Point to mask evaluation on 23 datasets.png" alt="drawing" width="1000"/> 

### 5.2 Zero-Shot Edge Detection
* Results:  
    <img src="./Zero-shot edge prediction on BSDS500.png" alt="drawing" width="400"/> <img src="./Zero-shot transfer to edge detection on BSDS500.png" alt="drawing" width="400"/>  
### 5.3 Zero-Shot Object Proposals

### 5.4 Zero-Shot Instance Segmentation
* Approach: 通过一个object detector(ViTDet)的输出框提示SAM;  
* Results:  
    <img src="./Instance segmentation results.png" alt="drawing" width="400"/> <img src="./analysis of instance segmentation results.png" alt="drawing" width="296"/>  


### 5.5 Zero-Shot Text-to-Mask
* Approach: 对每个面积大于100*100的mask, 提取CLIP的图像嵌入。在训练过程中, 用提取的CLIP图像嵌入作为SAM的第一次交互来提示SAM(因为CLIP的图像嵌入被训练为与文本嵌入对齐, 所以可以使用图像嵌入进行训练). 在推理时, 通过CLIP的文本编码器生成文本嵌入作为SAM的提示.  
* Results: SAM可以根据简单的文本提示进行分割. 当SAM无法仅从文本提示中分割正确的对象时, 增加额外的point prompt可以帮助SAM来分割。   
    <img src="./zero-shot text-to-mask.png" alt="drawing" width="400"/> 
    
### 5.2 Ablations
<img src="./ablation studies.png" alt="drawing" width="1000"/>   

## Q6. 文章的主要贡献是什么？
## A6.
1. 提出了一个用于通用图像分割的cv foundation model;
2. Data-centric(更侧重于提高数据质量和数量), 标注了一个包含1B maskde 数据集, 比现有的segment数据集大400倍;
3. 数据引擎: Assisted-manual -> Semi-automatic stage -> Fully automatic;  

## 7. 组合应用
1. [Grounding DINO + Segment Anything + Stable Diffusion](https://github.com/IDEA-Research/Grounded-Segment-Anything)
   <img src="./ground sam.png" alt="drawing" width="1000"/>  
