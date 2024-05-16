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
```python
import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from xmodelvlm.model.xmodelvlm import load_pretrained_model
from xmodelvlm.conversation import conv_templates, SeparatorStyle
from xmodelvlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from xmodelvlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def inference_once(args):

    disable_torch_init()
    model_name = args.model_path.split('/')[-1]
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)

    images = [Image.open(args.image_file).convert("RGB")]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # Input
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    # Inference
    with torch.inference_mode():
        start_time = time.time()
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
        end_time = time.time()
        execution_time = end_time-start_time
        print("the execution time (secend): ", execution_time)
    # Result-Decode
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    print(f"🚀 {model_name}: {outputs.strip()}\n")
if __name__ == '__main__':
   model_path = "Xmodel_VLM/xmodelvlm_1.1b" # 
   image_file = "assets/samples/demo.jpg"
   prompt_str = "Who is the author of this book?\nAnswer the question using a single word or phrase."
   # (or) What is the title of this book?
   # (or) Is this book related to Education & Teaching?
   
   args = type('Args', (), {
       "model_path": model_path,
       "image_file": image_file,
       "prompt": prompt_str,
       "conv_mode": "v1",
       "temperature": 0, 
       "top_p": None,
       "num_beams": 1,
       "max_new_tokens": 512,
       "load_8bit": False,
       "load_4bit": False,
   })()
   
   inference_once(args)
```

## 🪜 Step-by-step Tutorial

### Xmodel_VLM

#### 1️⃣ Prepare Xmodel_VLM checkpoints

#### 2️⃣ Prepare data

#### 3️⃣ Run everything with one click!


## 🤝 Acknowledgments


## ✏️ Reference


