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
pip install --upgrade pip
pip install -e ./llm-compressor -r requirements.txt
```

### Quantization

#### vllm pipeline:

```bash
python notebooks/do_compression.py \
                --device cuda \
                --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
                --dataset_name "wikitext" \
                --dataset_subset "wikitext-2-raw-v1" \
                --scheme "W8A8" \
                --targets "Linear" \
                --next_reg_lam 0.1 \
                --num_calibration_samples 512 \
                --max_seq_length 1024 \
                --seed 42 \
                --output_dir "quant_checkpoints" # output dir
```

Result quantized vllm checkpointed model saves in ```--output_dir # by default: quant_checkpoints```


### RAG launch

```bash
pip install "numpy<2" # downgrade numpy
python rag/demo_rag.py
```

* Input request in: ```rag/results/inp_query.txt```
* Output request in: ```rag/results/out_query.txt```
