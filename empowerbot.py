from flask import Flask, request, jsonify, render_template
from google.cloud import aiplatform
from sentence_transformers import SentenceTransformer

import openpyxl
from pinecone import Pinecone, ServerlessSpec # Example vector database, replace if needed
from langchain_text_splitters import MarkdownHeaderTextSplitter
from transformers import pipeline  # For summarization
import faiss
import numpy as np
from google.oauth2 import service_account

from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings

import openpyxl
import os

import time
import requests
from requests.exceptions import HTTPError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted



MODEL_ID = "gemini-pro"  # Replace with your actual endpoint ID
PROJECT_ID = "grantgpt-433615"
REGION = "us-central1"


# 0. AUTHENTICATE WITH GOOGLE CLOUD PLATFORM
def authicate_google():
    
    key_path = "C:/Users/a884470/backend/grantgp.json"
    
    credentials = service_account.Credentials.from_service_account_file(key_path)


    aiplatform.init(
        project=PROJECT_ID,
        credentials=credentials,
        location=REGION
    )
    print("Authenticated with Google Cloud Platform")


# 1. SET UP VERTEX AI ENVIRONMENT
authicate_google()

# 2. LOAD PINECONE MODEL FOR EMBEDDING
embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# 2a Define the embedding dimension (example: 384 for 'all-MiniLM-L6-v2')
embedding_dimension = embeddings.get_sentence_embedding_dimension()


# 3. INITIALIZE PINECONE VECTOR DATABASE
pc = Pinecone(api_key="pcsk_7RJrPF_3YGAhAWVnZ4zNTiU65AjkVE83kjJHborJTmLCGq13xt1xmred7smfcqrzGJQ1ME")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# 4. Initialize two separate indexes for full documents and summaries

index_name_full = "life-coaches-full"
index_name_summary = "life-coaches-summary"

if index_name_full not in pc.list_indexes().names():
    pc.create_index(
        name=index_name_full,
        dimension=embedding_dimension,
        metric="cosine",
        spec=spec
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name_full).status['ready']:
        time.sleep(1)


if index_name_summary not in pc.list_indexes().names():
    pc.create_index(
        name=index_name_summary,
        dimension=embedding_dimension,
        metric="cosine",
        spec=spec
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name_summary).status['ready']:
        time.sleep(1)

index_full = pc.Index(index_name_full)
time.sleep(1)
# view index stats
index_full.describe_index_stats()

index_summary = pc.Index(index_name_summary)
time.sleep(1)
# view index stats
index_summary.describe_index_stats()


# 5. Initialize summarization model
summarizer = pipeline("summarization")




# 6. LOAD PREVIOUSLY EMBEDDED KNOWLEDGE BASE
# (You could store these embeddings in Google Cloud Storage and load them here)
def load_knowledge_base():
  
   # Load the Excel workbook
    file_path = r"C:\Users\a884470\empowerbot-env\LifeCoaches1.xlsx"
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active

    # Iterate over coaches (starting from row 2)
    for row in range(2, sheet.max_row + 1):
        coach_name = sheet[f"A{row}"].value  # Get coach name from column A

        # Iterate over topics (starting from column 4, since A, B, and C are reserved)
        for col in range(4, sheet.max_column + 1):
            topic = sheet.cell(row=1, column=col).value  # Get topic from row 1
            if topic:  # Ensure there is a topic
                existing_teaching = sheet.cell(row=row, column=col).value
                
                # Only fill if it's empty
                if not existing_teaching:
                    print(f"Fetching teachings for {coach_name} on {topic}...")

                    # Get teachings from Vertex AI
                    teachings = generate_response(coach_name, topic)
                    #print(teachings)


                    # 4a. Generate embeddings for the full teachings
                    full_embedding = embeddings.encode(teachings)

                    # 4b. Generate a summary of the teachings
                    summary = summarizer(teachings, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                    print("I got to this point")

                    # 4c. Generate embeddings for the summary
                    summary_embedding = embeddings.encode(summary)


                    # 4d. Save both full and summary embeddings in the vector database
                    vector_id = f"{coach_name}_{topic}".replace(" ", "_")
                    

                    # Upsert full document embedding to the full index
                    upsert_with_notification(index_full,vector_id, full_embedding, "full")
                    

                    # Upsert summary embedding to the summary index
                    upsert_with_notification(index_summary,vector_id, summary_embedding, "summary")
                    
                    # Fill the teachings in the cell
                    sheet.cell(row=row, column=col).value = teachings

    # Save the updated Excel workbook
    wb.save(file_path)
    print("Excel sheet updated successfully.")
   
    return file_path

def upsert_with_notification(index, vector_id, embedding, index_type):
    """
    Upserts an embedding into the specified Pinecone index and notifies when successful.

    Parameters:
    - index (str): The  Pinecone index (either full or summary).
    - vector_id (str): Unique ID for the vector being upserted.
    - embedding (list): The embedding vector to be upserted.
    - index_type (str): Type of the index (either 'full' or 'summary') for notifications.
    """
    # Check if the vector ID already exists in the index
    existing_ids = index.fetch([vector_id]).ids
    if vector_id in existing_ids:
        print(f"Embedding for {vector_id} already exists in the {index_type} index. Skipping upsert.")
        return

    # Upsert the embedding into the specified index
    response = index.upsert([(vector_id, embedding)])

    # Check if the upsert was successful (based on Pinecone's response or wait time)
    if response:
        print(f"Embedding for {vector_id} successfully upserted into the {index_type} index!")
    else:
        print(f"Failed to upsert embedding for {vector_id} into the {index_type} index.")
    




#documents = load_knowledge_base()
#corpus_embeddings = embedding_model.encode(documents)

# Initialize FAISS Index
#index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
#index.add(np.array(corpus_embeddings))

# 7. FUNCTION TO RETRIEVE KNOWLEDGE FROM FAISS INDEX
#def retrieve(query, top_k=1):
    #query_embedding = embedding_model.encode([query])
    #distances, indices = index.search(np.array(query_embedding), top_k)
    #results = [(documents[i], distances[0][i]) for i in indices[0]]
    #return results[0][0]  # Return top result

# 8. SET UP VERTEX AI MODEL (FOR GENERATION)
# Function to generate text using Vertex AI model
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(ResourceExhausted))
def generate_response(coach_name, topic):
    input_text = f"Question: Tell me what you know about {coach_name}'s teachings on {topic}. This includes any advice, tips, or strategies."
    #response = generator_endpoint.predict(instances=[{"input": input_text}])
    # Fallback response for error handling
    empty_response = RunnableLambda(
        lambda x: AIMessage(content="Error processing document")
    )

    # Initialize the Vertex AI model client
    vertex_ai_client = VertexAI(
    temperature=1, model_name="gemini-1.0-pro", max_tokens=1024
    ).with_fallbacks([empty_response])

    # Define the input for the LLMChain
    var = {
       "text": input_text
    }

    time.sleep(30)
    # Create and run the LLMChain
    chain = LLMChain(
        llm=vertex_ai_client,
        prompt=PromptTemplate(
            input_variables=["text"],
            template=input_text
        )
    )
    response = chain.run(var)  
    #return response.predictions[0]
    
    return response




# 9. RAG PIPELINE
#def rag_pipeline(query):
    # Step 1: Retrieve relevant knowledge
    #retrieved_knowledge = retrieve(query)
    
    # Step 2: Generate a response using the retrieved knowledge
    #response = generate_response(query, retrieved_knowledge)
    #return response

# 10. FLASK ROUTE FOR CHATBOT
#@app.route("/chat", methods=["POST"])
#def chat():
    #query = request.json.get("query")
    
    # Run the RAG pipeline
    #response = rag_pipeline(query)
    
    #return jsonify({"response": response})


# 8. START THE FLASK APP
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load-knowledge-base', methods=['POST'])
def load_knowledge_base_route():
    try: 
        documents = load_knowledge_base()
        print("Knowledge base loaded:")
        return jsonify({"documents": documents})
    except HTTPError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


