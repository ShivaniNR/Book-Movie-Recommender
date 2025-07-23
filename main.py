import os
from langchain.llms import Ollama
from vector_space import EmbeddingManager
from retriever_tool import ToolManager
from chat_agent import ChatAgent
from topic_classifier import TopicClassifier

class Recommender:
    def __init__(self):
        # Initialize the ChatOpenAI model
        self.llm = Ollama(model='gemma3')

        # Initialize components
        book_vector_store_path = "data/book_vector_store"
        movie_vector_store_path = "data/movie_vector_store"
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = EmbeddingManager(embeddings_model_name).get_embeddings()
        
        self.tool_manager = ToolManager(self.llm, movie_vector_store_path, book_vector_store_path, embeddings)
        self.topic_classifier = TopicClassifier(self.llm)
        self.chat_agent = ChatAgent(self.llm, self.tool_manager)

    def get_chat_response(self, user_input):
        response = self.chat_agent.get_response(user_input, self.topic_classifier)
        return response['answer']