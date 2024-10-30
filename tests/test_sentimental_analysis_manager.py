import unittest
from managers.sentimental_analysis_manager import SentimentAnalysis

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalysis()

    # Test Positive Sentiment
    def test_analyze_sentiment_and_emotion_positive(self):
        text = "I love programming in Python. It's so versatile and powerful!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertEqual(result["sentiment"], "POSITIVE")
        self.assertGreater(result["sentiment_score"], 0)
        self.assertEqual(result["emotion"], "joy")
        self.assertGreater(result["emotion_score"], 0)

    # Test Negative Sentiment
    def test_analyze_sentiment_and_emotion_negative(self):
        text = "I hate waiting in long lines. It's so frustrating and annoying."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertEqual(result["sentiment"], "NEGATIVE")
        self.assertGreater(result["sentiment_score"], 0)
        self.assertEqual(result["emotion"], "anger")
        self.assertGreater(result["emotion_score"], 0)

    # Test Neutral Sentiment
    def test_analyze_sentiment_and_emotion_neutral(self):
        text = "The sky is blue and the grass is dark green."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertEqual(result["sentiment"], "NEGATIVE")
        self.assertGreater(result["sentiment_score"], 0)
        self.assertEqual(result["emotion"], "NEUTRAL")
        self.assertGreater(result["emotion_score"], 0)

    # Test Positive Sentiment using VADER
    def test_quick_sentiment_vader_positive(self):
        text = "I love programming in Python. It's so versatile and powerful!"
        result = self.analyzer.quick_sentiment_vader(text)
        self.assertIn("pos", result)
        self.assertIn("neg", result)
        self.assertIn("neu", result)
        self.assertIn("compound", result)
        self.assertGreater(result["pos"], 0)
        self.assertEqual(result["neg"], 0)
        self.assertGreater(result["neu"], 0)
        self.assertGreater(result["compound"], 0)

    # Test Negative Sentiment using VADER
    def test_quick_sentiment_vader_negative(self):
        text = "I hate waiting in long lines. It's so frustrating and annoying."
        result = self.analyzer.quick_sentiment_vader(text)
        self.assertIn("pos", result)
        self.assertIn("neg", result)
        self.assertIn("neu", result)
        self.assertIn("compound", result)
        self.assertEqual(result["pos"], 0)
        self.assertGreater(result["neg"], 0)
        self.assertGreater(result["neu"], 0)
        self.assertLess(result["compound"], 0)

    # Test Keyword Extraction
    def test_extract_keywords(self):
        text = "I love programming in Python. It's so versatile and powerful!"
        result = self.analyzer.extract_keywords(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("programming", result)
        self.assertIn("Python", result)
        self.assertIn("versatile", result)
        self.assertIn("powerful", result)

    # Test Engagement Analysis
    def test_analyze_engagement(self):
        text = "I love programming in Python! It's so versatile and powerful!"
        result = self.analyzer.analyze_engagement(text)
        self.assertIn("word_count", result)
        self.assertIn("sentence_count", result)
        self.assertIn("exclamation_count", result)
        self.assertEqual(result["word_count"], 10)
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["exclamation_count"], 2)

    # Test Full Analysis
    def test_full_analysis(self):
        text = "I love programming in Python. It's so versatile and powerful!"
        result = self.analyzer.full_analysis(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertIn("vader_scores", result)
        self.assertIn("keywords", result)
        self.assertIn("engagement_metrics", result)
        self.assertIsInstance(result["vader_scores"], dict)
        self.assertIsInstance(result["keywords"], list)
        self.assertIsInstance(result["engagement_metrics"], dict)

    # Test Empty Text Analysis
    def test_empty_text(self):
        text = ""
        result = self.analyzer.full_analysis(text)
        self.assertEqual(result["sentiment"], "POSITIVE")
        self.assertEqual(result["sentiment_score"], 0.0)
        self.assertEqual(result["emotion"], "NEUTRAL")
        self.assertEqual(result["emotion_score"], 0.0)
        self.assertEqual(result["vader_scores"]["compound"], 0.0)
        self.assertEqual(result["keywords"], [])
        self.assertEqual(result["engagement_metrics"]["word_count"], 0)

    # Test Long Positive Text
    def test_analyze_long_positive_text(self):
        text = "I really enjoy the way Python allows for clean coding. It makes life so much easier!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Long Negative Text
    def test_analyze_long_negative_text(self):
        text = "Waiting for the bus is the worst experience. It always seems to take forever!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "NEGATIVE")

    # Test Single Word Positive Emotion
    def test_single_word_positive(self):
        text = "Fantastic!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")
        self.assertEqual(result["emotion"], "joy")

    # Test Single Word Negative Emotion
    def test_single_word_negative(self):
        text = "I cannot believe this asshole!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "NEGATIVE")
        self.assertEqual(result["emotion"], "anger")

    # Test Sentence with Mixed Emotions
    def test_mixed_emotions(self):
        text = "I love the new features, but I hate the bugs!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Sentiment on an Exclamatory Statement
    def test_exclamatory_statement(self):
        text = "What an amazing day!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Sentiment on a Question
    def test_question_sentiment(self):
        text = "Isn't Python the best programming language?"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Neutral Statement
    def test_neutral_statement(self):
        text = "This is a  statement."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "NEUTRAL")

    # Test Mixed Sentiment using VADER
    def test_mixed_sentiment_vader(self):
        text = "I enjoy the features but dislike the price."
        result = self.analyzer.quick_sentiment_vader(text)
        self.assertIn("compound", result)

    # Test Simple Positive Statement
    def test_simple_positive_statement(self):
        text = "I am happy."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Simple Negative Statement
    def test_simple_negative_statement(self):
        text = "I am sad."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "NEGATIVE")

    # Test Punctuation Impact
    def test_punctuation_impact(self):
        text = "Wow!!! This is incredible!!!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Text with Special Characters
    def test_special_characters(self):
        text = "This is great! #awesome @Python"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Sentiment with Emoji
    def test_sentiment_with_emoji(self):
        text = "I love coding! 😊"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Uppercase Sentiment
    def test_uppercase_sentiment(self):
        text = "THIS IS AMAZING!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Sentence with Few Words
    def test_few_words(self):
        text = "Good job."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Long Negative Statement
    def test_long_negative_statement(self):
        text = "This is the worst service I have ever experienced, and I would not recommend it to anyone."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "NEGATIVE")

    # Test Analysis with Different Emotions
    def test_analysis_with_varied_emotions(self):
        text = "I feel excited about the future, but worried about the present."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Similar Words
    def test_similar_words_sentiment(self):
        text = "I'm thrilled with the results but disappointed by the process."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Repeated Words
    def test_repeated_words(self):
        text = "I really really like this!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Various Sentence Lengths
    def test_various_sentence_lengths(self):
        text = "Excellent service! Not what I expected."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Negative Text with Positive Ending
    def test_negative_text_with_positive_ending(self):
        text = "The weather is bad, but the sunset was beautiful."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Short Sentiment Analysis
    def test_short_sentiment_analysis(self):
        text = "Great!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Emotion on Exclamation
    def test_emotion_on_exclamation(self):
        text = "Hooray!!!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Analysis with Numbers
    def test_analysis_with_numbers(self):
        text = "I have 3 cats and they are amazing!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertEqual(result["sentiment"], "POSITIVE")

    # Test Analysis with Quotes
    def test_analysis_with_quotes(self):
        text = "'To be or not to be' is a profound statement."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Text with Parentheses
    def test_text_with_parentheses(self):
        text = "This is good (but could be better)."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Sentiment from External Sources
    def test_sentiment_external_source(self):
        text = "The movie was fantastic, but the ending was disappointing."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Complete Analysis with Mixed Inputs
    def test_complete_analysis_mixed_inputs(self):
        text = "It's a mixed bag. Some things are great, others not so much."
        result = self.analyzer.full_analysis(text)
        self.assertIn("sentiment", result)

    # Test Mixed Emotions and Sentiments
    def test_mixed_emotions_and_sentiments(self):
        text = "I love my job, but the commute is terrible."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Compound Sentences
    def test_compound_sentences(self):
        text = "I like coffee, but I prefer tea."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

    # Test Complex Sentences
    def test_complex_sentences(self):
        text = "Although I enjoy the outdoors, I also appreciate a good book."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)

# Run the tests
if __name__ == '__main__':
    unittest.main()
