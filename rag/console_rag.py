import argparse
from vectorizer import PDFVectorizer
import os
import sys
from datetime import datetime

class Color:
    """Simple color codes for terminal output"""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header():
    """Print a nice header"""
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}🤖 RAG Query System{Color.END}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*60}{Color.END}")

def print_step(message):
    """Print a step with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Color.YELLOW}[{timestamp}]{Color.END} {Color.BOLD}➜{Color.END} {message}")

def print_success(message):
    """Print success message"""
    print(f"{Color.GREEN}✓{Color.END} {message}")

def print_error(message):
    """Print error message"""
    print(f"{Color.RED}✗{Color.END} {message}")

def print_results(results, top_k):
    """Print search results in a formatted way"""
    print(f"\n{Color.BOLD}{Color.BLUE}📚 Top {top_k} Results:{Color.END}")
    print(f"{Color.BLUE}{'─'*60}{Color.END}")
    
    for i, doc in enumerate(results, 1):
        print(f"\n{Color.BOLD}Result #{i}:{Color.END}")
        print(f"{Color.BLUE}{'─'*40}{Color.END}")
        
        # Truncate long documents for display
        display_text = doc.strip()
        if len(display_text) > 300:
            display_text = display_text[:300] + "..."
        
        print(display_text)

def main():
    # Print header
    print_header()
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Retrieve and Generate answers from PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i input.txt -o output.txt
  %(prog)s -q "your question here"
  
Required: either -i or -q must be provided
        """
    )
    
    # Mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i", 
        help="Input file containing the query"
    )
    input_group.add_argument(
        "--query", "-q", 
        help="Direct query text (enclose in quotes)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="rag/results/out_query.txt",
        help="Output file path (default: rag/results/out_query.txt)"
    )
    parser.add_argument(
        "--index", "-x", 
        default="rag/vector_store",
        help="Path to vector index (default: rag/vector_store)"
    )
    parser.add_argument(
        "--top-k", "-k", 
        type=int, 
        default=3,
        help="Number of results to retrieve (default: 3)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed processing information"
    )
    
    args = parser.parse_args()
    
    try:
        # Get query text
        if args.input:
            print_step(f"Reading query from: {args.input}")
            try:
                with open(args.input, "r", encoding="utf-8") as f:
                    query = f.read().strip()
                print_success(f"Query loaded ({len(query)} characters)")
            except FileNotFoundError:
                print_error(f"Input file not found: {args.input}")
                sys.exit(1)
            except Exception as e:
                print_error(f"Error reading file: {e}")
                sys.exit(1)
        else:
            query = args.query.strip()
            print_step(f"Processing direct query")
            print_success(f"Query: {query[:50]}..." if len(query) > 50 else f"Query: {query}")
        
        # Display configuration
        print(f"\n{Color.BOLD}Configuration:{Color.END}")
        print(f"  Index path: {args.index}")
        print(f"  Output file: {args.output}")
        print(f"  Top K results: {args.top_k}")
        print(f"  Verbose mode: {'Yes' if args.verbose else 'No'}")
        
        # Initialize vectorizer
        print_step("Initializing vectorizer...")
        vectorizer = PDFVectorizer()
        
        # Load index
        print_step(f"Loading index from: {args.index}")
        if not os.path.exists(args.index):
            print_error(f"Index directory not found: {args.index}")
            sys.exit(1)
        
        vectorizer.load_index(args.index)
        print_success("Index loaded successfully")
        
        # Search
        print_step(f"Searching for: '{query[:30]}...'" if len(query) > 30 else f"Searching for: '{query}'")
        results = vectorizer.search(query, args.top_k)
        print_success(f"Found {len(results)} results")
        
        # Display results
        if args.verbose:
            print_results(results, args.top_k)
        
        # Format output for file
        output = f"Query: {query}\n\n"
        output += f"Retrieved {args.top_k} most relevant results:\n\n"
        output += "=" * 60 + "\n\n"
        
        for i, doc in enumerate(results, 1):
            output += f"Result #{i}:\n"
            output += "-" * 40 + "\n"
            output += doc.strip() + "\n\n"
        
        output += "=" * 60 + "\n\n"
        output += f"Answer to query: {query}\n"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save to file
        print_step(f"Saving results to: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        
        print_success(f"Results saved successfully ({os.path.getsize(args.output)} bytes)")
        
        # Final summary
        print(f"\n{Color.BOLD}{Color.GREEN}{'='*60}{Color.END}")
        print(f"{Color.BOLD}{Color.GREEN}✅ Processing Complete!{Color.END}")
        print(f"{Color.BOLD}{Color.GREEN}{'='*60}{Color.END}")
        print(f"Query processed: {query[:50]}..." if len(query) > 50 else f"Query processed: {query}")
        print(f"Results saved to: {args.output}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}⚠️  Process interrupted by user{Color.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()