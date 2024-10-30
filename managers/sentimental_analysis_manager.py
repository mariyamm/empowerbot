import re
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import pandas as pd
from typing import Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure that the NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class SentimentAnalysis:
    def __init__(self):
        # Initialize models
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_sentiment_and_emotion(self, text: str) -> Dict[str, float]:
        # Sentiment Analysis using transformers
        sentiment_result = self.sentiment_analyzer(text)[0]
        sentiment = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        # Emotion Analysis using transformers
        emotion_result = self.emotion_analyzer(text)
        print("Emotion Result Structure:", emotion_result)  # Print the result structure for debugging
        
        # Assuming the expected structure is still a list of dictionaries
        if isinstance(emotion_result, list) and len(emotion_result) > 0 and isinstance(emotion_result[0], list):
            emotion = emotion_result[0][0]['label']
            emotion_score = emotion_result[0][0]['score']
        else:
            emotion = "Unknown"
            emotion_score = 0.0
         # Print the emotion and score for debugging
        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "emotion": emotion,
            "emotion_score": emotion_score
        }

    
    def quick_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Quickly analyzes sentiment using VADER for real-time response.
        Returns a dictionary with scores for positive, negative, neutral, and compound sentiments.
        """
        vader_scores = self.vader_analyzer.polarity_scores(text)
        return vader_scores
    
    def extract_keywords(self, text: str) -> list:
        """
        Extracts keywords from the text using spaCy's NLP model.
        Filters for nouns, verbs, and adjectives.
        """

        # Tokenize the text into words
        words = word_tokenize(text)
        
        # Define stop words
        stop_words = set(stopwords.words('english'))
        
        # Filter out stop words and non-alphabetic words
        keywords = [word for word in words if word.lower() not in stop_words and word.isalpha()]
        
        return keywords
    
    def analyze_engagement(self, text: str) -> Dict[str, int]:
        """
        Analyzes engagement level based on sentence length and punctuation (e.g., exclamation marks).
        Returns word count, sentence count, and number of exclamation marks.
        """
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]', text))
        exclamation_count = text.count("!")
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "exclamation_count": exclamation_count
        }
    
    def full_analysis(self, text: str) -> Dict[str, any]:
        """
        Combines all analysis methods into one comprehensive analysis.
        Returns a dictionary with sentiment, emotion, VADER scores, keywords, and engagement metrics.
        """
        # Get sentiment and emotion analysis
        sentiment_emotion = self.analyze_sentiment_and_emotion(text)
        
        # Get quick sentiment analysis with VADER
        vader_scores = self.quick_sentiment_vader(text)
        
        # Extract keywords
        keywords = self.extract_keywords(text)
        
        # Analyze engagement
        engagement = self.analyze_engagement(text)
        
        # Compile all results into a single dictionary
        return {
            "sentiment": sentiment_emotion['sentiment'],
            "sentiment_score": sentiment_emotion['sentiment_score'],
            "emotion": sentiment_emotion['emotion'],
            "emotion_score": sentiment_emotion['emotion_score'],
            "vader_scores": vader_scores,
            "keywords": keywords,
            "engagement_metrics": engagement
        }

