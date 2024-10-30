from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import pipeline  # For summarization
from requests.exceptions import HTTPError
from datetime import datetime
from managers.google_cloud_auth import GoogleCloudAuth
from managers.pinecone_manager import PineconeManager
from managers.knowledge_base_manager import KnowledgeBaseManager
from managers.chatbot import Chatbot
from managers.user_profile_manager import UserProfileManager
from managers.user_assesment_manager import UserAssesmentManager
from managers.sentimental_analysis_manager import SentimentAnalysis
import time


# Configuration Constants
MODEL_ID = "gemini-pro"
PROJECT_ID = "grantgpt-433615"
REGION = "us-central1"
KEY_PATH = "C:/Users/a884470/backend/grantgp.json"
EXCEL_FILE_PATH = r"C:\Users\a884470\empowerbot-env\LifeCoaches.xlsx"
INITIAL_QUESTIONS = [
    "What's your name?",
    "What are your main goals for coaching?",
    "Any specific areas you'd like help with (e.g., career, personal growth)?"
]

PINECONE_API_KEY = "pcsk_7RJrPF_3YGAhAWVnZ4zNTiU65AjkVE83kjJHborJTmLCGq13xt1xmred7smfcqrzGJQ1ME"


# FLASK ROUTE FOR LOADING KNOWLEDGE BASE
app = Flask(__name__)

# Initialize components
authenticator = GoogleCloudAuth(PROJECT_ID, REGION, KEY_PATH)
authenticator.authenticate()

#pinecone_manager = PineconeManager(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_manager = PineconeManager(PINECONE_API_KEY)

embeddings = SentenceTransformer('all-MiniLM-L6-v2')

summarizer = pipeline("summarization")

knowledge_manager = KnowledgeBaseManager(embeddings, summarizer, pinecone_manager)

chatbot = Chatbot(pinecone_manager, embeddings)

user_profile_manager = UserProfileManager(embeddings, pinecone_manager)


# Define a route to the homepage
@app.route('/')
def index():
    return render_template('index.html')


# Define a route to load the knowledge base
@app.route('/load-knowledge-base', methods=['POST'])
def load_knowledge_base_route():
    try: 
        documents = knowledge_manager.load_knowledge_base()
        print("Knowledge base loaded successfully.")
        return jsonify({"documents": documents})
    except HTTPError as e:
        return jsonify({"error": str(e)}), 404
    
# Route to start the chat
@app.route("/start", methods=["POST"])
def start_chat():
    data = request.get_json()
    phone_number = data.get("phone_number")

    if not phone_number:
        return jsonify({"error": "Phone number is required"}), 400

    # Check if user profile exists
    user_profile = user_profile_manager.retrieve_user_profile(phone_number)

    # If profile exists, greet user and return to chat
    if user_profile:
        return jsonify({
            "message":  f"Welcome back, {user_profile['name']}!",
            "new_user": False
            })
    else: 
        print("New user detected, sending initial questions.")
        # Return the first question
        return jsonify({
            "message": "Hello! Thank you for reaching out. Before we begin, I will need your approval to collect and store your responses to the following questions. Your information will be kept confidential and used only for the purpose of providing coaching services. Are you ready to proceed? (Yes/No)",
            "new_user": True,
        })


# Collect initial data and create user profile
@app.route("/collect_initial_data", methods=["POST"])
def collect_initial_data():
    try:
        # Parse JSON data from request
        data = request.get_json()

        # Extract data fields
        phone_number = data.get("phone_number", "")
        approval = data.get("approval", "")
        name = data.get("name", "")
        goals = data.get("goals", "")
        focus_area = data.get("focus_area", "")

        # Construct metadata
        user_metadata = {
            "phone_number": phone_number,
            "approval": approval,
            "name": name,
            "goals": goals,
            "focus_area": focus_area,
            "timestamp": datetime.now().isoformat()
        }
        print("Received user metadata:", user_metadata)

        # Save user profile - ensure `save_user_profile` is set up to accept metadata without a vector if needed
        user_profile_manager.create_user_profile(phone_number, user_metadata)

        # Send success response
        return jsonify({"message": f"Thanks, {name}! We’re ready to start."}), 200

    except Exception as e:
        print("Error in collect_initial_data:", e)
        return jsonify({"error": "An error occurred while processing data."}), 500



# Endpoint for continuing the chat with the bot after initialization
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    phone_number = data.get('phone_number')
    question = data.get('question')

    if not phone_number or not question:
        return jsonify({"error": "Phone number and question are required"}), 400

    # Process the question and generate a response
    response = chatbot.generate_rag_response(question)

    # Save the chat history
    return jsonify({"message": response})




    
if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000, debug=True)
    analyzer = SentimentAnalysis()

    texts = [
       
        "I love the new design of the website! It's so user-friendly and visually appealing.",
        "The customer service was terrible. I had to wait for over an hour to get a response.",
        "This product is amazing! It exceeded all my expectations.",
        "I'm not satisfied with the quality of the material. It feels cheap and flimsy.",
        "The movie was fantastic! The plot was gripping and the acting was top-notch.",
        "I had a bad experience at the restaurant. The food was cold and the service was slow.",
        "The book was a great read. I couldn't put it down.",
        "I'm disappointed with the recent update. It introduced more bugs than it fixed.",
        "The concert was an unforgettable experience. The band played all my favorite songs."
    ]

    print("Enter text for sentiment analysis (type 'exit' to quit):")
    while True:
        text = input("Text: ")
        if text.lower() == 'exit':
            break
        result = analyzer.full_analysis(text)
        print(result)
        print("\n")  # Add a newline for better readability

