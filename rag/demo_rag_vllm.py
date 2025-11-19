from vectorizer import PDFVectorizerVLLM


if __name__ == "__main__":
    # Initialize with your compressed model
    vectorizer = PDFVectorizerVLLM(
        model_path="/home/buka2004/QUANT-RAG-MIPT/quant_checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0/wikitext/next_reg_lam=0.1"
    )

    # Create index with PDFs
    vectorizer.create_index(
        pdf_directory="/home/buka2004/QUANT-RAG-MIPT/rag/vector_store",
        save_dir="/home/buka2004/QUANT-RAG-MIPT/rag/results",
        chunk_size=256,
        overlap=64
    )

    # Simple search
    results = vectorizer.search("How Adam works???", k=5)
    for i, result in enumerate(results):
        print(f"Result {i+1} (score: {result['score']:.4f}): {result['text']}")

    # RAG search with answer generation
    rag_result = vectorizer.rag_search("your question here", k=3)
    print(f"Answer: {rag_result['answer']}")
    print(f"Context used: {rag_result['context']}")