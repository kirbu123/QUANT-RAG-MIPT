import argparse
from vectorizer import PDFVectorizer
import os

def main():
    parser = argparse.ArgumentParser(description="RAG Query System")
    
    # Mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", help="Input file with query")
    input_group.add_argument("--query", "-q", help="Direct query text")
    
    parser.add_argument("--output", "-o", default="rag/results/out_query.txt",
                       help="Output file path")
    parser.add_argument("--index", "-x", default="rag/vector_store",
                       help="Vector index path")
    parser.add_argument("--top-k", "-k", type=int, default=3,
                       help="Number of results")
    
    args = parser.parse_args()
    
    # Get query text
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            query = f.read()
    else:
        query = args.query
    
    print(f"Query: {query}")
    
    # Initialize and search
    vectorizer = PDFVectorizer()
    vectorizer.load_index(args.index)
    results = vectorizer.search(query, args.top_k)
    
    # Format and save output
    output = f"Query: {query}\n\nUsing {args.top_k} results:\n\n"
    for i, doc in enumerate(results, 1):
        output += f"[Result {i}]\n{doc}\n\n"
    
    output += f"Answer to query: {query}\n"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()