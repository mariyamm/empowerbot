
#This class will be used to manage the user personality assesment

class UserAssesmentManager:

    def __init__(self):
        # Initialize trait scores for Big Five
        self.big_five_scores = {"openness": 0, "conscientiousness": 0, "extraversion": 0, "agreeableness": 0, "neuroticism": 0}
        self.big_five_questions = {
            "openness": ["I enjoy trying new things.", "I am open to different perspectives."],
            "conscientiousness": ["I pay attention to details.", "I like to stay organized."],
            "extraversion": ["I am outgoing and sociable.", "I feel energized by social interactions."],
            "agreeableness": ["I am empathetic toward others.", "I am a helpful person."],
            "neuroticism": ["I often feel stressed or anxious.", "I get upset easily."]
        }
        self.big_five_answers = {trait: [] for trait in self.big_five_questions}

        # Initialize MBTI preferences and questions
        self.mbti_scores = {"E-I": 0, "S-N": 0, "T-F": 0, "J-P": 0}
        self.mbti_questions = {
            "E-I": ["I enjoy social gatherings.", "I prefer time alone to recharge."],
            "S-N": ["I trust facts over theories.", "I like exploring abstract concepts."],
            "T-F": ["I prioritize logic over emotions.", "I make decisions based on feelings."],
            "J-P": ["I like having a set plan.", "I prefer to be flexible and spontaneous."]
        }
        self.mbti_answers = {dimension: [] for dimension in self.mbti_questions}

        # Initialize Enneagram scores and questions
        self.enneagram_scores = {f"type_{i}": 0 for i in range(1, 10)}
        self.enneagram_questions = {
            "type_1": ["I strive for perfection in all things.", "I have high standards for myself and others."],
            "type_2": ["I like helping others and feel fulfilled by it.", "I put others' needs above my own."],
            "type_3": ["I am driven to achieve success.", "I am focused on my image and achievements."],
            "type_4": ["I feel unique and different from others.", "I am in touch with my emotions."],
            "type_5": ["I enjoy spending time alone with my thoughts.", "I am an analytical person."],
            "type_6": ["I value security and safety.", "I am cautious and plan ahead."],
            "type_7": ["I seek new and exciting experiences.", "I am an optimistic person."],
            "type_8": ["I am assertive and decisive.", "I stand up for myself and others."],
            "type_9": ["I prefer to avoid conflict.", "I value peace and harmony."]
        }
        self.enneagram_answers = {type_key: [] for type_key in self.enneagram_questions}

    ### BIG FIVE METHODS
    def get_big_five_question(self, trait, question_index):
        """
        Returns the question text for the specified Big Five trait and question index.
        """
        return self.big_five_questions[trait][question_index]

    def assess_big_five_response(self, trait, score):
        """
        Collects and stores the user's score for a specific Big Five trait question.
        """
        self.big_five_answers[trait].append(score)
        if len(self.big_five_answers[trait]) == len(self.big_five_questions[trait]):
            # Calculate the average score for the trait after all questions are answered
            self.big_five_scores[trait] = sum(self.big_five_answers[trait]) / len(self.big_five_answers[trait])

    ### MBTI METHODS
    def get_mbti_question(self, dimension, question_index):
        """
        Returns the question text for the specified MBTI dimension and question index.
        """
        return self.mbti_questions[dimension][question_index]

    def assess_mbti_response(self, dimension, score):
        """
        Collects and stores the user's score for a specific MBTI dimension question.
        """
        self.mbti_answers[dimension].append(score)
        if len(self.mbti_answers[dimension]) == len(self.mbti_questions[dimension]):
            # Calculate the sum score for the dimension after all questions are answered
            self.mbti_scores[dimension] = sum(self.mbti_answers[dimension])

    def get_mbti_type(self):
        """
        Generates the MBTI type code based on dimension scores.
        """
        mbti_type = (
            "E" if self.mbti_scores["E-I"] >= 0 else "I",
            "S" if self.mbti_scores["S-N"] >= 0 else "N",
            "T" if self.mbti_scores["T-F"] >= 0 else "F",
            "J" if self.mbti_scores["J-P"] >= 0 else "P"
        )
        return "".join(mbti_type)

    ### ENNEAGRAM METHODS
    def get_enneagram_question(self, type_key, question_index):
        """
        Returns the question text for the specified Enneagram type and question index.
        """
        return self.enneagram_questions[type_key][question_index]

    def assess_enneagram_response(self, type_key, score):
        """
        Collects and stores the user's score for a specific Enneagram type question.
        """
        self.enneagram_answers[type_key].append(score)
        if len(self.enneagram_answers[type_key]) == len(self.enneagram_questions[type_key]):
            # Calculate the average score for the type after all questions are answered
            self.enneagram_scores[type_key] = sum(self.enneagram_answers[type_key]) / len(self.enneagram_answers[type_key])

    def get_enneagram_type(self):
        """
        Determines the highest scoring Enneagram type as the dominant type.
        """
        return max(self.enneagram_scores, key=self.enneagram_scores.get)

    ### FINAL RESULTS
    def get_results(self):
        """
        Compiles all results for the assessments.
        """
        return {
            "Big Five": self.big_five_scores,
            "MBTI": self.get_mbti_type(),
            "Enneagram": self.get_enneagram_type()
        }
