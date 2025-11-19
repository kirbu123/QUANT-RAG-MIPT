# QUANT-RAG-MIPT

Post-Training Quantization for Large Language Models - MIPT Research Project

## Project Overview

This project implements and experiments with Post-Training Quantization (PTQ) techniques for Large Language Models, combining DeepSpeed, vllm and SmoothQuant methodologies. We apply our quantization techniques to RAG of pdf articles.


We invite you to our: [literary review](https://docs.google.com/spreadsheets/d/1vHBZKW7IKO7Z1W8Cb-9dAWTvs5KyeQz7na4ITVk3UbE/edit?usp=sharing) && [presentation](https://docs.google.com/presentation/d/1stMcldEc-rVStRlNXcyGW1V6gGGSOP-P7vhGS15sQJk/edit?usp=sharing)


### Clone the repository

```bash
git clone https://github.com/kirbu123/QUANT-RAG-MIPT.git
cd QUANT-RAG-MIPT
```

### Setup enviroment

```bash
# recomended python version = 3.8 for deepspeed and 3.10 for vllm && rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd llm-compressor && pip install -e .
```

### Quantization

#### vllm pipeline:

```bash
python notebooks/do_compression.py \
                --device cuda \ # device name
                --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \ # LLM model name
                --dataset_name "wikitext" \ # train dataset name
                --dataset_subset "wikitext-2-raw-v1" \ # train dataset subset
                --scheme "W8A8" \ # quantization weight/activation scheme
                --targets "Linear" \
                --next_reg_lam 0.1 \ # my custom next layer regularization coef
                --num_calibration_samples 512 \
                --max_seq_length 1024 \ # sequence lenght
                --seed 42 \ # random seed
                --output_dir "vllm_out" # output dir
```

Result quantized vllm checkpointed model saves in ```--output_dir```


### RAG launch

```bash
python rag/demo_rag.py
```

* Input request in: ```rag/inp_query.txt```
* Output request in: ```rag/out_query.txt```
