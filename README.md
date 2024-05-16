<h1 align="center">
Xmodel_VLM: A Simple Baseline for Multimodal Vision Language Model
</h1>

<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Xiaoduo%20HuggingFace-blue.svg)](https://huggingface.co/XiaoduoAILab/Xmodel_VLM)
[![arXiv](https://img.shields.io/badge/Arxiv-2405.09215-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.09215) 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/XiaoduoAILab/XmodelVLM.git)[![github](https://img.shields.io/github/stars/XiaoduoAILab/XmodelVLM.svg?style=social)](https://github.com/XiaoduoAILab/XmodelVLM.git)  


</h5>





## 🛠️ Install

1. Clone this repository and navigate to MobileVLM folder
   ```bash
   git clone [https://github.com/](https://github.com/XiaoduoAILab/XmodelVLM.git)
   cd xmodelvlm
   ```

2. Install Package
    ```Shell
    conda create -n xmodelvlm python=3.10 -y
    conda activate xmodelvlm
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## 🗝️ Quick Start

#### Example for Xmodel_VLM model inference
```bash
python inference.py
```

## 🪜 Step-by-step Tutorial

### Xmodel_VLM
The overall architecture of our network, closely mirrors that of LLaVA-1.5. It consists of three key components: 
* a vision encoder (CLIP ViT-L/14)
* a lightweight languagemodel (LLM)
* a projector responsible for aligning the visual and textual spaces (XDP)
  
Refer to [our paper](https://arxiv.org/pdf/2405.09215) for more details!  
![assets/model archtecture.jpeg](https://github.com/XiaoduoAILab/XmodelVLM/blob/main/assets/model%20archtecture.jpeg)  
![assets/XDP.jpeg](https://github.com/XiaoduoAILab/XmodelVLM/blob/main/assets/XDP.jpeg)




The training process of Xmodel_VLM is divided into two stages: 

- stage I: pre-training
  - ❄️ frozen vision encoder + 🔥 **learnable** XDP projector + ❄️ **learnable** LLM
- stage II: multi-task training
  - ❄️ frozen vision encoder + 🔥 **learnable** XDP projector + 🔥 **learnable** LLM
![https://github.com/XiaoduoAILab/XmodelVLM/tree/main/assets/training strategy.jpeg](https://github.com/XiaoduoAILab/XmodelVLM/blob/main/assets/training%20strategy.jpeg)




#### 1️⃣ Prepare Xmodel_VLM checkpoints
Please firstly download MobileLLaMA chatbot checkpoints from [huggingface website](https://huggingface.co/XiaoduoAILab/Xmodel_VLM)
#### 2️⃣ Prepare data

#### 3️⃣ Run everything with one click!


## 🤝 Acknowledgments


## ✏️ Reference


