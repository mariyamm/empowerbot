import pinecone
import uuid

class UserProfileManager:
    def __init__(self, embeddings, pinecone_manager):
        self.embeddings = embeddings
        self.pinecone_manager = pinecone_manager
        self.index_name = "user-profiles"
        self.index = self.pinecone_manager.create_index(self.index_name, self.embeddings.get_sentence_embedding_dimension())

    def create_user_profile(self, phone_number, user_data):
        """
        Creates a new user profile and stores the associated embeddings in Pinecone.
        
        Args:
            user_data (dict): A dictionary containing user-specific data, e.g., {'name': 'John', 'interests': 'psychology'}.
        
        Returns:
            str: The generated user ID for the new profile.
        """
        #user_id = str(uuid.uuid4())  # Generate a unique ID for the user
        profile_id = f"profile_{phone_number}"
        user_embedding = self.generate_user_embedding(user_data)
        self.upsert_user_embedding(profile_id, user_embedding)
        print(f"User profile for {phone_number} saved.")
        return profile_id

    def generate_user_embedding(self, user_data):
        """
        Generates an embedding vector from user data by encoding it.
        
        Args:
            user_data (dict): A dictionary of user attributes to encode, e.g., {'name': 'John', 'interests': 'psychology'}.
        
        Returns:
            list: A list representing the embedding vector.
        """
        user_data_str = " ".join(f"{k}: {v}" for k, v in user_data.items())
        user_embedding = self.embeddings.encode(user_data_str)
        return user_embedding

    def upsert_user_embedding(self, user_id, embedding):
        """
        Inserts or updates a user's embedding in the Pinecone index.
        
        Args:
            user_id (str): The unique ID of the user.
            embedding (list): The embedding vector to store.
        """
        response = self.index.upsert([(user_id, embedding)])
        print(f"User profile upsert for {user_id}: {'successful' if response else 'failed'}")

    def retrieve_user_profile(self, user_id):
        """
        Retrieves a user's profile from the Pinecone index based on user ID.
        
        Args:
            user_id (str): The unique ID of the user.
        
        Returns:
            dict: The user profile data, if found.
        """
        try:
            response = self.index.fetch([user_id])
            if 'vectors' in response and user_id in response['vectors']:
                return response['vectors'][user_id]
        except Exception as e:
            print("Error during fetch:", e)

        print("Profile not found or fetch error.")
        return None

    def search_similar_profiles(self, user_data, top_k=5):
        """
        Searches for profiles similar to the given user data.
        
        Args:
            user_data (dict): A dictionary of user attributes for similarity search.
            top_k (int): Number of similar profiles to return.
        
        Returns:
            list: A list of similar user profiles.
        """
        user_embedding = self.generate_user_embedding(user_data)
        results = self.index.query(queries=[user_embedding], top_k=top_k)
        return results.get('matches', [])
    


