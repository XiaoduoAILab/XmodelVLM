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
   git clone https://github.com/XiaoduoAILab/XmodelVLM.git
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

- prepare benchmark data
  - We evaluate models on a diverse set of 9 benchmarks, *i.e.* GQA, MMBench, MMBench-cn, MME, POPE, SQA, TextVQA, VizWiz, MM-Vet.  For example, you should follow these instructions to manage the datasets:
  - <details>
    <summary> Data Download Instructions </summary>

    - download some useful [data/scripts](https://github.com/Meituan-AutoML/MobileVLM/releases/download/v0.1/benchmark_data.zip) pre-collected by us.
      - `unzip benchmark_data.zip && cd benchmark_data`
      - `bmk_dir=${work_dir}/data/benchmark_data`
    - gqa
      - download its image data following the official instructions [here](https://cs.stanford.edu/people/dorarad/gqa/download.html)
      - `cd ${bmk_dir}/gqa && ln -s /path/to/gqa/images images`
    - mme
      - download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
      - `cd ${bmk_dir}/mme && ln -s /path/to/MME/MME_Benchmark_release_version images`
    - pope
      - download coco from POPE following the official instructions [here](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco).
      - `cd ${bmk_dir}/pope && ln -s /path/to/pope/coco coco && ln -s /path/to/coco/val2014 val2014`
    - sqa
      - download images from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
      - `cd ${bmk_dir}/sqa && ln -s /path/to/sqa/images images`
    - textvqa
      - download images following the instructions [here](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip).
      - `cd ${bmk_dir}/textvqa && ln -s /path/to/textvqa/train_images train_images`
    - mmbench
      - no action is needed.

    </details>


#### 3️⃣ Run everything with one click!
We provide detailed pre-training, fine-tuning and testing shell scripts, for example:
```shell
bash scripts/pretrain.sh 0,1,2,3  #GPU:0,1,2,3
```

## 🤝 Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA): Thanks for their wonderful work! 👏
- [MobileVLM](https://github.com/Meituan-AutoML/MobileVLM): Thanks for their wonderful work! 👏
  
## ✏️ Reference

If you find Xmodel_VLM useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:

```
@misc{xu2024xmodelvlm,
      title={Xmodel-VLM: A Simple Baseline for Multimodal Vision Language Model}, 
      author={Wanting Xu and Yang Liu and Langping He and Xucheng Huang and Ling Jiang},
      year={2024},
      eprint={2405.09215},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
