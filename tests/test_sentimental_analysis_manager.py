import unittest
from managers.sentimental_analysis_manager import SentimentAnalysis

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalysis()

    def test_analyze_sentiment_and_emotion_positive(self):
        text = "I love programming in Python. It's so versatile and powerful!"
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertEqual(result["sentiment"], "pos")
        self.assertGreater(result["sentiment_score"], 0)
        self.assertEqual(result["emotion"], "joy")
        self.assertGreater(result["emotion_score"], 0)

    def test_analyze_sentiment_and_emotion_negative(self):
        text = "I hate waiting in long lines. It's so frustrating and annoying."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertEqual(result["sentiment"], "neg")
        self.assertGreater(result["sentiment_score"], 0)
        self.assertEqual(result["emotion"], "anger")
        self.assertGreater(result["emotion_score"], 0)

    def test_analyze_sentiment_and_emotion_neutral(self):
        text = "The sky is blue and the grass is green."
        result = self.analyzer.analyze_sentiment_and_emotion(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertEqual(result["sentiment"], "neu")
        self.assertGreater(result["sentiment_score"], 0)
        self.assertEqual(result["emotion"], "neutral")
        self.assertGreater(result["emotion_score"], 0)

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

    def test_extract_keywords(self):
        text = "I love programming in Python. It's so versatile and powerful!"
        result = self.analyzer.extract_keywords(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("programming", result)
        self.assertIn("Python", result)
        self.assertIn("versatile", result)
        self.assertIn("powerful", result)

    def test_analyze_engagement(self):
        text = "I love programming in Python! It's so versatile and powerful!"
        result = self.analyzer.analyze_engagement(text)
        self.assertIn("word_count", result)
        self.assertIn("sentence_count", result)
        self.assertIn("exclamation_count", result)
        self.assertEqual(result["word_count"], 9)
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["exclamation_count"], 2)

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

    def test_empty_text(self):
        text = ""
        result = self.analyzer.full_analysis(text)
        self.assertEqual(result["sentiment"], "neu")
        self.assertEqual(result["sentiment_score"], 0.0)
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["emotion_score"], 0.0)
        self.assertEqual(result["vader_scores"]["compound"], 0.0)
        self.assertEqual(result["keywords"], [])
        self.assertEqual(result["engagement_metrics"]["word_count"], 0)

    def test_non_english_text(self):
        text = "Je t'aime programmer en Python. C'est tellement polyvalent et puissant!"
        result = self.analyzer.full_analysis(text)
        self.assertIn("sentiment", result)
        self.assertIn("sentiment_score", result)
        self.assertIn("emotion", result)
        self.assertIn("emotion_score", result)
        self.assertIn("vader_scores", result)
        self.assertIn("keywords", result)


#run the tests  
if __name__ == '__main__':
    unittest.main()