# Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification(ECCV2023, Amazon Web Services)
[paper with code](https://paperswithcode.com/paper/tip-adapter-training-free-adaption-of-clip)  

## Q1. 文章针对的问题？
### A1. 利用CLIP预训练模型实现few-shot classification  

## Q2. 文章要验证的假设是什么？
### A2. 在不增加参数的情况下既保留CLIP的training-free能力, 又达到好的few-shot classification效果  

## Q3. 有哪些相关研究？如何归类？
### A3. 
(1) Data-efficient Transfer Learning: MoCo, BYOL, VirTex, CLIP, DeCLIP, ALIGN, CoOp, CLIP-Adapter, WiSE-FT;  
(2) Transformer;  
(3) Cache Model: Cache Model将训练图像的feature及其label存储为key-value数据库. 在推理过程中, 将测试图像的feature视为query, 检索key-value数据库, 类似于transformer中的注意力机制。  


## Q4. 文章的解决方案是什么？
### A4.
<img src="./the pipeline of Tip-Adapter.png" alt="drawing" width="600"/> 

#### 4.1 Training-free CLIP-Adapter
给定预训练的CLIP模型和N类(每类K个样本)的数据集, 目标是对N类进行图像分类:  
1. Cache Model Construction: 对每个训练图像数据, 通过CLIP的visual encoder来提取feature(C维向量, L2归一化)得到Ftrain, 并将其ground-truth转换为N维one-hot编码, 得到Ltrain.
   <img src="./cache model construction.png" alt="drawing" width="400"/>  
2. Tip-Adapter: 
   (1) 由于key和query是L2归一化的，因此1−ftest\*Ftrain相当于计算ftest和训练图像特征Ftrain之间的余弦距离。采用指数函数将距离转换为非负值A, β用来调节锐度。通过A\*Ltrain获得从Cache Model检索到的值;
   ```
   A = exp(−β(1−ftest*Ftrain))
   ```
   (2) predicted logits:
   ```
   logits = α*A*Ltrain + Wc*ftest
   其中:
   (1) Wc表示CLIP的文本分类器, 通过CLIP的text encoder得到的每个类别文本的text feature;
   (2) α代表残差比率;  
   ```

3. CLIP-adapter和Tip-adapter之间的差异可以总结如下:  
(1) CLIP-adapter随机初始化W1和W2，并通过SGD学习它们. Tip-adapter直接将W1设置为缓存的训练特征Ftrain，将W2设置为地面实况标签Ltrain的one-hot编码, 这是非参数的，不需要训练.  
(2) Tip-adapter的bottleneck dimension等于NK, 而CLIP-adapter选择low-dimensional bottleneck以防止过拟合.这表明, 通过合适的初始化, 在few-shot数据集上的过拟合问题得到了很大缓解, 这进一步释放了高维线性层的拟合能力.    
(3) Tip-adapter引入了一个新的激活函数A=exp(−β(1−ftest*Ftrain)), 由于其输入是归一化特征空间中的距离, 因此它的边界在0和1之间。CLIP-adapter是用ReLU来处理无边界输入。在新激活的帮助下，计算出的距离可以很好地调节，并有助于提高性能.

4. Tip-adapter可以将Cache Model作为一个很好的初始化点, 并通过SGD继续微调Tip-adapter的W1＝Ftrain权重, 以绕过随机初始化的CLIP适配器。

## Q5. 评估数据集是什么？评估方法是什么？
### A5. ImageNet, StandfordCars, UCF101, Caltech101, Flowers102, SUN397, DTD, EuroSAT, FGVCAircraft, OxfordPets, and Food101. 

## Q6. 文章的实验是怎么设计的？
### A6.
#### 6.1 Comparison on ImageNet
<img src="./Classification acc on ImageNet.png" alt="drawing" width="350"/>  
<img src="./Performance of different backbone on ImageNet.png" alt="drawing" width="450"/>  

<img src="./Classification acc of models under both CoOp-style and CLIP-style pre-processing on ImageNet.png" alt="drawing" width="350"/>  
<img src="./Fine-tuning time and classification accuracy of 16-shot on ImageNet.png" alt="drawing" width="450"/>  

#### 6.2 Performances on Other Datasets
<img src="./Classification acc on other 10 datasets.png" alt="drawing" width="1000"/>


#### 6.3 Ablation Studies
<img src="./ablation studies on Tip-Adapter.png" alt="drawing" width="500"/>

## 参考文献
[1] Frozen clip models are efficient video learners.  
[2] Parameter-efficient transfer learning for nlp.  