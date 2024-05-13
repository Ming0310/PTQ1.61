# PTQ1.61


## Contents
- [Install](#install)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Install
```
conda create -n ptq161 python=3.10 -y
conda activate ptq161
git clone https://github.com/zjq0455/PTQ1.61.git
cd PTQ1.61
pip install --upgrade pip 
pip install -e .
```

## Usage
**We provide full script to run PTQ1.61 in `./scripts/`**. We use LLaMa-7B as an example here:
1. Obtain the channel-wise scales required for initialization:
we offer the script that you can generate channel-wise scales and shifts by yourself:
```
python generate_act_scale_shift.py --model /PATH/TO/LLaMA/llama-7b
```

2. Weight-only quantization
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/LLAMA/llama-7b --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--ckpt_path /CHECKPOINT/TO/FIRST/PTQ \
--calib_dataset wikitext2  \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

3. Compensate with LoRA
```
cd qat
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py --model_id /PATH/TO/LLAMA/llama-7b \
--ckpt /CHECKPOINT/TO/FIRST/PTQ --lora_r 64 -s 20000
```
5. Merge with LoRA
```
CUDA_VISIBLE_DEVICES=0 python test_perplexity.py  --model_path /PATH/TO/LLAMA/llama-7b \
--ckpt /CHECKPOINT/TO/FIRST/PTQ \
--lora_path ./outputs/CHECKPOINT_NAME/20000-64 \
--output_path /PATH/TO/MERGED/MODEL
```
6. Second PTQ
```
CUDA_VISIBLE_DEVICES=0 python main.py --model /PATH/TO/MERGED/MODEL --epochs 20 --output_dir ./log/llama-7b --eval_ppl --wbits 4 --abits 16 --quant_type mix --lwc \
--ckpt_path /CHECKPOINT/TO/SECOND/PTQ \
--calib_dataset wikitext2  \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

More detailed and optional arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--abits`: activation quantization bits.
- `--group_size`: group size of weight quantization. If no set, use per-channel quantization for weight as default.
- `--lwc`: activate the Learnable Weight Clipping (LWC).
- `--let`: activate the Learnable Equivalent Transformation (LET).
- `--lwc_lr`: learning rate of LWC parameters, 1e-2 as default.
- `--let_lr`: learning rate of LET parameters, 5e-3 as default.
- `--epochs`: training epochs. You can set it as 0 to evaluate pre-trained OmniQuant checkpoints.
- `--nsamples`: number of calibration samples, 128 as default.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--tasks`: evaluating zero-shot tasks.
- `--resume`: loading pre-trained OmniQuant parameters.
- `--multigpu`: to inference larger network on multiple GPUs
- `--real_quant`: real quantization, which can see memory reduce
- `--save_dir`: saving the quantization model for further exploration.

## Results


## Related Project
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137)

## Citation
If you use our PTQ1.61 approach in your research, please cite our paper:
```

```
