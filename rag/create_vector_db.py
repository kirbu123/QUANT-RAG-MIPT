from vectorizer import PDFVectorizer
from datasets import load_dataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", default="")
    parser.add_argument("--chunk_size", default=256, type=int)
    parser.add_argument("--overlap", default=64, type=int)
    parser.add_argument("--save_path", default="rag/vector_store")
    opt = parser.parse_args()

    vectorizer = PDFVectorizer()
    vectorizer.create_index("C:/Users/user/Desktop/rag/rag_small_data", opt.save_path, opt.chunk_size, opt.overlap)