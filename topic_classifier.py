class TopicClassifier:
    def __init__(self, llm):
        self.llm = llm
        self.topics = ["movies", "books", "others"]
    
    def classify(self, query):
        prompt = f"Classify the following question into one of these topics: '{','.join(self.topics)}': '{query}'"
        response = self.llm.predict(text = prompt, max_tokens=10)
        topic = response.strip().lower()
        return topic