from vectorizer import PDFVectorizer
import os

if __name__ == "__main__":
    
    with open("rag/inp_query.txt", "r") as f:
        query = f.read()

    print(f"User query: {query}")

    vectorizer = PDFVectorizer()
    vectorizer.load_index("rag/vector_store")
    results = vectorizer.search(query, 3)

    output = "Using following context:\n"
    for doc in results:
        output += "\n" + doc + "\n"
    output += f"\nAnswer on query: \n{query}"

    with open("rag/out_query.txt", "w", encoding="utf-8") as f:
        f.write(output)

    # print(f"RAG output: \n {output}")    