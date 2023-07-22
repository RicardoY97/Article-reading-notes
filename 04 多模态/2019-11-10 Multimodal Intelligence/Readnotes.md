# Multimodal Intelligence: Representation Learning, Information Fusion, and Applications(2019, 京东)
[pdf](./Multimodal%20Intelligence%20Representation%20Learning%20Information%20Fusion%20and%20Applications.pdf)

## FUSION

### A. Simple Operation-based Fusion
1. concatenation
2. weighted sum: 要求各模态的feature同维度;
3. 使用NAS技术搜索适当层做fusion

### B. Attention-based Fusion
#### B1. Image attention  
1. Visual7W: Grounded Question Answering in Images(VQA任务, 2015)     
   <img src="./image%20attention%20in%20visual7w.png" alt="drawing" width="300"/>
   <img src="./image%20attention%20in%20visual7w%202.png" alt="drawing" width="575"/>
2. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention(image caption, 2015)  
3. Where to look: Focus regions for visual question answering(VQA任务, 2016)   
   将来自不同区域的视觉特征[v1, v2, ...]和文本查询q映射到一个共享空间中，然后在该空间中通过计算内积来衡量两种模态之间的相关性，然后加权求和  
   <img src="./image%20attention%20in%20where%20to%20look.png" alt="drawing" width="700"/>  
4. Stacked attention networks for image question answering(VQA任务, 2016)  
   <img src="./image%20attention%20in%20SANs.png" alt="drawing" width="500"/>  
   <img src="./image%20attention%20in%20SANs%202.png" alt="drawing" width="500"/>
   <img src="./image%20attention%20in%20SANs%203.png" alt="drawing" width="337"/>  
5. Ask, attend and answer: Exploring questionguided spatial attention for visual question answering(VQA任务, 2016)  
   <img src="./image%20attention%20in%20SMN.png" alt="drawing" width="500"/>  
6. Dynamic memory networks for visual and textual question answering(VQA任务, 2016)  
7. Bottom-Up and Top-Down Attention for Image Captioningand Visual Question Answering()  
   自上而下: 由非视觉或任务特定情境驱动的注意机制(由LSTM实现);  
   自下而上: 纯视觉前馈注意机制(由Faster R-CNN 实现, 提取一组显著的图像区域，每个区域由一个合并的卷积特征向量表示);  
   <img src="./image%20attention%20in%20Bottom-up%20and%20Top-down.png" alt="drawing" width="500"/> 
8. Co-attending freeform regions and detections with multi-modal multiplicative feature embedding for visual question answering(VQA,2018)  
   free-form: 全图特征的attention(例如，对于“今天天气怎么样？”这个问题，天空中可能不存在检测框);
   detection-box: 局部区域的attention(例如，对于“你看到什么动物？”的问题，显示的检测区域会更简单);
   <img src="./image%20attention%20in%20Co-attending.png" alt="drawing" width="700"/> 
   
   
#### B2. Image and text co-attention
1. Hierarchical question-image co-attention for visual question answering

## Q1. 论文针对的问题？
### A1. 

## Q2. 文章要验证的假设是什么？
### A2. 

## Q3. 有哪些相关研究？如何归类？
### A3. 

## Q4. 文章的解决方案是什么？关键点是什么？
### A4. 

## Q5. 评估数据集是什么？评估方法是什么？
### A5. 

## Q6. 文章的实验是怎么设计的？
### A6. 


## Q7. 实验方法和结果能不能支持文章提出的假设？
### A7. 

## Q8. 文章的主要贡献是什么？
### A8. 

## Q9. 是否存在不足或者问题？
### A9. 


## Q10. 下一步还可以继续的研究方向是什么？  
### A10. 