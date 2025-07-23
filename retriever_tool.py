from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.agents import Tool

class ToolManager:
    def __init__(self, llm, movies_vector_path, books_vector_path, embeddings):
        self.llm = llm
        self.movies_vector_path = movies_vector_path
        self.books_vector_path = books_vector_path
        self.embeddings = embeddings
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self):
        #Load FAISS vector stores for movies and books
        movies_vector_store = FAISS.load_local(self.movies_vector_path, self.embeddings, allow_dangerous_deserialization=True)
        books_vector_store = FAISS.load_local(self.books_vector_path, self.embeddings, allow_dangerous_deserialization=True)


        #Define prompt templates for the tools
        prompt_template = """If the context is not relevant, 
        please answer the question by using your own knowledge about the topic
        
        {context}
        
        Question: {question}
        """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Initialize RetrievalQA tools with the FAISS vector stores
        movies_qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=movies_vector_store.as_retriever(search_kwargs={"k": 3}), chain_type_kwargs={"prompt": PROMPT})
        books_qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=books_vector_store.as_retriever(search_kwargs={"k": 3}), chain_type_kwargs={"prompt": PROMPT})

        # Return a dictionary of tools for movies and books
        return {
            "movies": Tool(name="MoviesTool", func=movies_qa.run, description="Retrieve movie information."),
            "books": Tool(name="BooksTool", func=books_qa.run, description="Retrieve book information.")
        }

    def get_tool(self, topic):
        return self.tools.get(topic)