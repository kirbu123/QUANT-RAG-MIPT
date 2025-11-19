from vectorizer import PDFVectorizer

if __name__ == "__main__":

    inp_path = "rag/results/inp_query.txt"
    out_path = "rag/results/out_query.txt"
    
    with open(inp_path, "r") as f:
        query = f.read()

    print(f"User query: {query}")

    vectorizer = PDFVectorizer()
    vectorizer.load_index("rag/vector_store")
    results = vectorizer.search(query, 3)

    output = "Using following context:\n"
    for doc in results:
        output += "\n" + doc + "\n"
    output += f"\nAnswer on query: \n{query}"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
