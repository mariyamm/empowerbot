from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
from langchain_google_vertexai import VertexAI
from langchain.schema import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
from managers.pinecone_manager import PineconeManager


class Chatbot:
    def __init__(self, pinecone_manager, embeddings, model_id="gemini-pro", temperature=1, max_tokens=1024):
        self.pinecone_manager = pinecone_manager
        self.embeddings = embeddings
        self.vertex_ai_client = VertexAI(temperature=temperature, model_name=model_id, max_tokens=max_tokens)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(ResourceExhausted))
    def generate_response(self, prompt):
        empty_response = RunnableLambda(lambda x: AIMessage(content="Error processing document"))
        vertex_ai_client_with_fallback = self.vertex_ai_client.with_fallbacks([empty_response])
        var = {"text": prompt}
        time.sleep(30)
        chain = LLMChain(
            llm=vertex_ai_client_with_fallback,
            prompt=PromptTemplate(input_variables=["text"], template=prompt)
        )
        return chain.run(var)
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(ResourceExhausted))
    def generate_rag_response(self, question: str) -> str:
        """Generates a response using Retrieval-Augmented Generation (RAG) approach."""
        # Encode the query
        query_embedding = self.embeddings.encode(question).tolist()

        # Retrieve context from Pinecone
        index = self.pinecone_manager.get_index("life-coaches-full")
        retrievals = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        # Collect context from matches
        context_parts = [
            match.get("metadata", {}).get("text", "[No relevant text found]") for match in retrievals["matches"]
        ]
        context = " ".join(context_parts)

        # Generate response based on retrieved context
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        response = self.generate_response(prompt)

        return response
