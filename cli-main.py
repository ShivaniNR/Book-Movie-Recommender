import os
from langchain.llms import Ollama
from vector_space import EmbeddingManager
from retriever_tool import ToolManager
from chat_agent import ChatAgent
from topic_classifier import TopicClassifier

#if different paid llms are used, import them here
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_API_KEY')
# OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.0'))

# Initialize the ChatOpenAI model
llm = Ollama(model='gemma3')


# Initialize components
book_vector_store_path = "data/book_vector_store"
movie_vector_store_path = "data/movie_vector_store"
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = EmbeddingManager(embeddings_model_name).get_embeddings()
tool_manager = ToolManager(llm, movie_vector_store_path, book_vector_store_path,embeddings)
topic_classifier = TopicClassifier(llm)
chat_agent = ChatAgent(llm, tool_manager)


print("Chatbot is ready to talk! Type 'quit' to exit.")
    
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    response = chat_agent.get_response(user_input, topic_classifier)
    print(f"You: {user_input}")
    print(f"Chatbot: {response['answer']}")
