# QUANT-RAG-MIPT

Post-Training Quantization for Large Language Models - MIPT Research Project

## Project Overview

This project implements and experiments with Post-Training Quantization (PTQ) techniques for Large Language Models, combining DeepSpeed, vllm and SmoothQuant methodologies.

We invite you to our: [literary review](https://docs.google.com/spreadsheets/d/1vHBZKW7IKO7Z1W8Cb-9dAWTvs5KyeQz7na4ITVk3UbE/edit?usp=sharing)

### RAG

We apply our quantization techniques to RAG of pdf articles.


### Clone the repository

```bash
git clone https://github.com/kirbu123/QUANT-RAG-MIPT.git
cd QUANT-RAG-MIPT
```

### Clone the repository

```bash
# recomended python version = 3.8 for deepspeed and 3.10 for vllm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

