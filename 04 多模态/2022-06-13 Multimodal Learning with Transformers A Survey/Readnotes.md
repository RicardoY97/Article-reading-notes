# Multimodal Learning with Transformers: A Survey(2022)
[pdf](./Multimodal%20Learning%20with%20Transformers%20A%20Survey.pdf)

## 1. **Tokenization and token embedding of different modal inputs for Transformers**
<img src="./Tokenization%20and%20token%20embedding%20comparison%20for%20multi-modal%20inputs%20for%20Transformers.png" alt="drawing" width="700"/>  

## 2. Special/customized tokens
<img src="./Special%20or%20customized%20tokens.png" alt="drawing" width="400"/>  

## 3. **Transformer-based cross-modal interactions**
*在多模态Transformer中, 不同模态间的交互(fusion, align)本质上是由 self-attention或者cross-Attention处理的.*  
1. Early Summation: 在早期将不同模态的token叠加;<sup><a href="#ref1">[1,2]</a></sup> 
2. Early Concatenation: 在早期将不同模态的token拼接;<sup><a href="#ref1">[3,4,5,6]</a></sup>  
3. Hierarchical Attention (multi-stream to one-stream): 先不同模态分别建模, 后期拼接进行多模态交互;<sup><a href="#ref1">[7]</a></sup>  
4. Hierarchical Attention (one-stream to multi-stream): 先拼接进行多模态交互, 后期不同模态分别建模;<sup><a href="#ref1">[8]</a></sup>
5. Cross-Attention: 不同模态数据互相作为Query embedding, 进行多模态交互;<sup><a href="#ref1">[9,10]</a></sup>
6. Cross-Attention to Concatenation: 在Cross-Attention之后再拼接融合;<sup><a href="#ref1">[11,12,13]</a></sup>

<img src="./Transformer-based%20cross-modal%20interactions.png" alt="drawing" width="700"/>   
<img src="./Self-attention%20variants%20for%20multi-modal%20interaction%20or%20fusion.png" alt="drawing" width="700"/> 

## 4. **Transformers for Multimodal Pretraining**
### 4.1 Task-Agnostic Multimodal Pretraining
*基于Transformer的多模态预训练, 关键是驱动Transformer(encoder w/, w/o decoder)学习跨模态交互*  
1. MLM: 掩码语言建模;  
2. MRR: 掩码图像区域预测/分类、掩蔽区域回归;  
3. VLM: 视觉-语言匹配;  
4. ITM: 图像文本匹配;
5. PRA: 短语-区域对齐;  
6. WRA: 单词-区域对齐;  
7. VSM: 视频字幕匹配;  
8. MFM: 掩码帧建模;  
9. FOM: 帧顺序建模;  
10. NSP: 下一句预测;  
11. MSG: 掩码句子生成;
12. MGM: 掩码组建模;
13. PrefixLM: 前缀语言建模;  
14. 视频条件掩蔽语言模型;  
15. 文本条件掩蔽帧模型;  
16. VTLM: 视觉翻译语言建模;  
17. 图像条件掩蔽语言建模

<img src="./Pretext%20task%20comparison%20of%20multi-modal%20pretraining%20Transformer%20models.png" alt="drawing" width="700"/>   

### 4.2 Task-Specific Multimodal Pretraining

## 参考文献
[1] Actortransformers for group activity recognition;  
[2] A large long-term person reidentification benchmark with clothes change;  
[3] Videobert: A joint model for video and language representation learning;  
[4] Graphcodebert: Pre-training code representations with data flow;  
[5] Learning audio-visual speech representation by masked multimodal cluster prediction;  
[6] Fused acoustic and text encoding for multimodal bilingual pretraining and speech translation;  
[7] Ai choreographer: Music conditioned 3d dance generation with aist++;  
[8] Interbert: Vision-and-language interaction for multi-modal pretraining;  
[9] Vilbert: Pretraining taskagnostic visiolinguistic representations for vision-and-language tasks;  
[10] Pano-avqa: Grounded audio-visual question answering on 360deg videos;  
[11] Humor knowledge enriched transformer for understanding multimodal humor;  
[12] Product1m: Towards weakly supervised instance-level product retrieval via cross-modal pretraining;  
[13] Multimodal transformer for unaligned multimodal language sequences;  