import os
import re
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
import aiohttp
import PyPDF2
import io
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.max_retries = 3
        self.retry_delay = 2
        
    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text content from PDF bytes
        """
        try:
            text = ""
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"PDF text extraction failed: {str(e)}")
    
    async def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with overlap for context preservation
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            # Try to break at sentence end if possible
            if end < len(text):
                sentence_enders = re.finditer(r'[.!?]\s+', text[start:end])
                positions = [m.end() + start for m in sentence_enders]
                if positions:
                    end = positions[-1]  # Break at last sentence end in chunk
            
            chunks.append(text[start:end].strip())
            
            # Move start position, accounting for overlap
            start = end - overlap
            
            if start >= len(text):
                break
                
        return chunks
    
    async def generate_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks using Gemini API
        """
        embeddings = []
        
        async with aiohttp.ClientSession() as session:
            for chunk in text_chunks:
                for attempt in range(self.max_retries):
                    try:
                        url = f"{self.base_url}/models/embedding-001:embedContent?key={self.api_key}"
                        payload = {
                            "model": "models/embedding-001",
                            "content": {
                                "parts": [{"text": chunk}]
                            }
                        }
                        
                        async with session.post(url, json=payload) as response:
                            if response.status == 200:
                                data = await response.json()
                                embeddings.append(data["embedding"]["values"])
                                break
                            elif response.status == 429:
                                logger.warning(f"Rate limited, retrying in {self.retry_delay} seconds...")
                                await asyncio.sleep(self.retry_delay * (attempt + 1))
                            else:
                                error_text = await response.text()
                                logger.error(f"Embedding generation failed: {error_text}")
                                if attempt == self.max_retries - 1:
                                    raise Exception(f"Embedding generation failed after {self.max_retries} attempts")
                    except Exception as e:
                        logger.error(f"Error generating embedding: {str(e)}")
                        if attempt == self.max_retries - 1:
                            raise
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return embeddings
    
    async def process_pdf_document(self, pdf_content: bytes, client_id: str, document_id: str) -> Dict:
        """
        Process a PDF document: extract text, chunk it, and generate embeddings
        Returns the processed chunks and embeddings for storage in Pinecone
        """
        try:
            # Extract text from PDF
            text = await self.extract_text_from_pdf(pdf_content)
            
            if not text.strip():
                raise Exception("No text could be extracted from the PDF")
            
            # Chunk the text
            chunks = await self.chunk_text(text)
            
            # Generate embeddings for each chunk
            embeddings = await self.generate_embeddings(chunks)
            
            # Prepare data for Pinecone storage
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{client_id}_{document_id}_{i}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "client_id": client_id,
                        "document_id": document_id,
                        "chunk_index": i,
                        "text": chunk,
                        "processed_at": datetime.utcnow().isoformat()
                    }
                })
            
            return {
                "success": True,
                "vectors": vectors,
                "chunk_count": len(chunks),
                "total_text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_response(self, query: str, context: List[str], conversation_history: List[Dict] = None) -> Dict:
        """
        Generate a response using Gemini API based on query and context
        """
        # Prepare the context for the prompt
        context_text = "\n\n".join(context) if context else "No relevant context found."
        
        # Prepare conversation history if provided
        history_text = ""
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"
        
        # Create the prompt with context and instructions
        prompt = f"""
        You are a helpful assistant for a business. Use the following context information to answer the user's question.
        If the answer cannot be found in the context, politely say so. Don't make up information.
        
        Context information:
        {context_text}
        
        Conversation history:
        {history_text}
        
        User question: {query}
        
        Assistant:
        """
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.base_url}/models/gemini-pro:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "temperature": 0.2,
                            "topK": 40,
                            "topP": 0.8,
                            "maxOutputTokens": 1024
                        }
                    }
                    
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                            
                            return {
                                "success": True,
                                "response": response_text,
                                "prompt_length": len(prompt),
                                "response_length": len(response_text)
                            }
                        elif response.status == 429:
                            logger.warning(f"Rate limited, retrying in {self.retry_delay} seconds...")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                        else:
                            error_text = await response.text()
                            logger.error(f"Response generation failed: {error_text}")
                            if attempt == self.max_retries - 1:
                                return {
                                    "success": False,
                                    "error": f"API error: {error_text}"
                                }
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e)
                    }
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def validate_api_key(self) -> bool:
        """
        Validate the Gemini API key
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/models?key={self.api_key}"
                async with session.get(url) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available Gemini models
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/models?key={self.api_key}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"] for model in data["models"]]
                    else:
                        logger.error(f"Failed to fetch models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            return []

# Global instance
gemini_service = GeminiService()
