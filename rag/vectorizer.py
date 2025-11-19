"""
Class to vectorize PDF files
Class to vectorize PDF files using VLLM with compressed Llama model
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader 
import pickle
import os
from vllm import LLM
from vllm.sampling_params import SamplingParams


class PDFVectorizerVLLM:
    def __init__(self, model_path: str = "/home/buka2004/QUANT-RAG-MIPT/quant_checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0/wikitext/next_reg_lam=0.1"):
        """
        Class to construct vector DB from PDF files using VLLM with compressed Llama model

        Parameters:
            model_path (str): path to compressed model checkpoint for VLLM
        """

        print(f"[info] Loading compressed model from: {model_path}")
        
        # Initialize VLLM model
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=2048
        )
        
        self.index = None
        self.documents = []
        
    def _get_embeddings(self, texts, normalize_embeddings=True):
        """Get embeddings using VLLM model"""
        # Use VLLM's embedding capability
        embeddings = self.llm.encode(texts)
        
        # Convert to numpy and normalize if required
        embeddings_np = embeddings.cpu().numpy()
        
        if normalize_embeddings:
            # Normalize each embedding
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            embeddings_np = embeddings_np / norms
            
        return embeddings_np

    def pdf2text(self, pdf_path: str):
        """Extract text from PDF file"""

        text = ""

        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Can't read text from {pdf_path}")
            print(e)
            return ""
        
        return text

    def chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 64):
        """Divide text on chunks"""

        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def create_index(self, 
                     pdf_directory: str, 
                     save_dir: str = "vector_store_vllm",
                     chunk_size=256,
                     overlap=64):
        """Create and save vector index using VLLM embeddings"""
        all_chunks = []
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            text = self.pdf2text(pdf_path)
            if text.strip():
                chunks = self.chunk_text(text, chunk_size, overlap)
                all_chunks.extend(chunks)
                print(f"[info] Processed {pdf_file}: got {len(chunks)} chunks")
        
        self.documents = all_chunks
        print(f"[info] Total chunks: {len(self.documents)}")
        
        # make embeddings using VLLM
        embeddings = self._get_embeddings(self.documents, normalize_embeddings=True)
        
        # create faiss index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        # save index and embeddings
        self.save_index(save_dir, embeddings)
        print(f"[info] Save vector index in: {save_dir}")

    def save_index(self, save_dir: str, embeddings: np.ndarray):
        """Save vector index and meta"""
        os.makedirs(save_dir, exist_ok=True)
        
        # save faiss index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        
        # save docs
        with open(os.path.join(save_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # save embeddings
        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
         
        # Save model info
        with open(os.path.join(save_dir, "model_info.txt"), 'w') as f:
            f.write("vllm_compressed_llama")

    def load_index(self, load_dir: str = "vector_store_vllm"):
        """Load saved index"""

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Can't find {load_dir}")
        
        # load faiss index
        self.index = faiss.read_index(os.path.join(load_dir, "faiss.index"))
        
        # load documents
        with open(os.path.join(load_dir, "documents.pkl"), 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"[info] Download index from: {load_dir}")
        print(f"[info] Uploaded {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5):
        """Search `k` nearest to query documents"""

        if self.index is None:
            raise ValueError("Index is not created. Run create_index method")
        
        query_embedding = self._get_embeddings([query], normalize_embeddings=True)

        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'text': self.documents[idx],
                'score': scores[0][i]
            })
        
        return results

    def generate_answer(self, query: str, context: str, max_tokens: int = 256):
        """Generate answer using VLLM with RAG context"""
        
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {query}

Answer:"""
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            stop=["\n\n"]
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        
        return outputs[0].outputs[0].text

    def rag_search(self, query: str, k: int = 3, max_tokens: int = 256):
        """Perform RAG search and generate answer"""
        
        # Search for relevant documents
        search_results = self.search(query, k=k)
        
        # Combine context from search results
        context = "\n".join([result['text'] for result in search_results])
        
        # Generate answer using the context
        answer = self.generate_answer(query, context, max_tokens)
        
        return {
            'answer': answer,
            'context': context,
            'search_results': search_results
        }


class PDFVectorizer:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Class to construct vector DB from PDF files

        Parameters:

            model (str): name of vector embeding model. Default is `sentence-transformers/all-MiniLM-L6-v2`. 
        """

        self.model = SentenceTransformer(model)
        self.index = None
        self.documents = []
        
    def pdf2text(self, pdf_path: str):
        """Extract text from PDF file"""

        text = ""

        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Can't read text from {pdf_path}")
            print(e)
            return ""
        
        return text

    def chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 64):
        """Divide text on chunks"""

        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def create_index(self, 
                     pdf_directory: str, 
                     save_dir: str = "vector_store",
                     chunk_size=256,
                     overlap=64):
        """Create and save vector index"""
        all_chunks = []
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            text = self.pdf2text(pdf_path)
            if text.strip():
                chunks = self.chunk_text(text, chunk_size, overlap)
                all_chunks.extend(chunks)
                print(f"[info] Processed {pdf_file}: got {len(chunks)} chunks")
        
        self.documents = all_chunks
        print(f"[info] Total chunks: {len(self.documents)}")
        
        # make embedings
        embeddings = self.model.encode(self.documents, normalize_embeddings=True)
        
        # create faiss index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        # save index and embedings
        self.save_index(save_dir, embeddings)
        print(f"[info] Save vector index in: {save_dir}")

    def save_index(self, save_dir: str, embeddings: np.ndarray):
        """Save vector index and meta"""
        os.makedirs(save_dir, exist_ok=True)
        
        # save faiss index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss.index"))
        
        # save docs
        with open(os.path.join(save_dir, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # save embedings
        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
         
        # with open(os.path.join(save_dir, "model_info.txt"), 'w') as f:
        #     f.write(self.model.get_sentence_embedding_dimension().__str__())

    def load_index(self, load_dir: str = "vector_store"):
        """Load saved index"""

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Can't find {load_dir}")
        
        # load faiss index
        self.index = faiss.read_index(os.path.join(load_dir, "faiss.index"))
        
        # load documents
        with open(os.path.join(load_dir, "documents.pkl"), 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"[info] Download index from: {load_dir}")
        print(f"[info] Uploaded {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 5):
        """Search `k` nearest to query documents"""

        if self.index is None:
            raise ValueError("Index is not created. Run create_index method")
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx in indices[0]:
            results.append(self.documents[idx])
        
        return results
