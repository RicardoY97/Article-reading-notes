# Visual Instruction Tuning(2023, Microsoft Research)
[paper with code](https://paperswithcode.com/paper/visual-instruction-tuning)  
[demo](https://llava.hliu.cc/)  

### ***keywords***
* Visual Instruction Tuning;
* Large Language and Vision Assistant;
  

## 1. 数据pipeline: GPT-assisted Visual Instruction Data Generation
step1: 准备一个image-text pair的数据集, 比如CC或者LAION;  
step2: 利用纯语言的GPT-4或ChatGPT作为强有力的老师, 创建包含视觉内容的instruction-following数据;
   ```
   作者使用了两种类型的符号表示：
   (1)caption: caption从不同的角度描述视觉场景。
   (2)bbox: 边界框描述了object类别及其空间位置。
   符号表示的作用是让图像能够转换成LLM可识别序列, 才能被纯语言的GPT-4或ChatGPT使用。
   ```
step3: 建立三种类型的instruction: 对于每个type, 需要首先创建几个例子, 在上下文学习中用作查询GPT-4的seed examples。
   * Conversation: 设计类似助理和人之间的对话。对图像的视觉内容提出了一系列不同的问题，包括object的类型、数量、动作、位置、相对位置。答案的语气就像助理看到图像并回答问题一样(问题的答案必须要明确)。prompt的生成可以看附录的Table 10;
   * Detailed description: 对于图像更细致的描述。从预设的列表中随机抽取问题, 让GPT-4来生成答案;
   * Complex reasoning: 以上两种类型侧重于视觉内容本身，在此基础上进一步创造更深入的推理问题。答案通常需要一个循序渐进的推理过程，遵循严格的逻辑。

结果: 总计158K的样本, 包括58K的Conversation、23K的detailed description和77k的Complex reasoning.

<img src="./an example to illustrate the instruction-following data.jpg" alt="drawing" width="600"/> 

## 2. Visual Instruction Tuning   
<img src="./LLaVA network architecture.jpg" alt="drawing" width="400"/> 

### 2.1 Vision Encoder
* model: pre-trained CLIP visual encoder ViT-L/14;
* feature: 最后一层transformer layer前后两个特征;
* project: Linear layer, 映射到和word embedding一样的dim;

### 2.2 Training
* data: 对于每个图像输入, 组织多回合对话数据[(q1, a1), (q2, a2), ..., (qT, aT)], at视为是assistant的响应. 第t轮的指令以如下形式得到:  
  <img src="./instruct at t-th turn.jpg" alt="drawing" width="600"/> 

* target: 通过LLM原始的自回归任务预测prediction tokens(图中绿色的部分):    
  <img src="./input squence used to train model.jpg" alt="drawing" width="600"/> 

* 训练过程包括两个stage:  
  - Stage 1: Pre-training for Feature Alignment. 在这个stage里只训练project参数, LLM和vision encoder都被冻住. 使用的数据是Conversation type, 即对简单的描述性问题, 输出对应的image caption;
  - Stage 2: Fine-tuning End-to-End. 在这个stage里只冻住vision encoder, 训练LLM和project。分为聊天机器人和科学问答两种场景:
   ```
   Multimodal Chatbot: conversation是多轮的, 其他两种是单轮。在训练中统一采样;
   Science QA: 将数据组织为单回合对话，将问题和上下文组织为Xinstruct, 将推理和答案组织为Xa;
   ```

## 3. 效果
### 3.1 Multimodal Chatbot
以GPT-4的结果作为评估标准:  
<img src="./result of chatbot.jpg" alt="drawing" width="600"/>   
从下面的示例中可以看出, LLaVa已经可以理解图像, 并且发现其中发现违和的部分或者联想:  
<img src="./example of chatbot.jpg" alt="drawing" width="600"/>   
<img src="./Example prompt demonstrating LLaVA and GPT-4’s visual input capability.jpg" alt="drawing" width="600"/>

在我们自己数据上的效果:    
<img src="./example2 of chatbot.jpg" alt="drawing" width="600"/>

<img src="./example4 of chatbot.jpg" alt="drawing" width="600"/>

<img src="./example3 of chatbot.jpg" alt="drawing" width="600"/>




### 3.2 ScienceQA
<img src="./result of science QA.jpg" alt="drawing" width="600"/>