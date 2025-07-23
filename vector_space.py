import json
from pathlib import Path
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm

class EmbeddingManager:
    def __init__(self, model_name):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def get_embeddings(self):
        return self.embeddings

class VectorSpaceManager:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager
        self.embeddings = embedding_manager.get_embeddings()
    
    def create_vector_store(self, documents):
        vector_store = FAISS.from_documents(
            documents=documents[:2],
            embedding=self.embeddings
        )

        # Create the vector store in batches to avoid memory issues and to speed up the process
        with tqdm(total=len(documents), desc="Creating vector store") as pbar:
            batch_size = 100
            for i in range(2, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                temp_vector_store = FAISS.from_documents(batch, self.embeddings)    # Temporary vector store for the current batch
                vector_store.merge_from(temp_vector_store)                          # Merge the current batch into the main vector store
                pbar.update(len(batch))
        
        return vector_store

    def save_vector_space(self, vector_store, save_path):
        print(f"Saving vector space to {save_path}...")
        vector_store.save_local(save_path)
        print(f"Finished!")
    
    def load_vector_space(self, save_path):
        print(f"Lodaing vector space from {save_path}")
        return FAISS.load_local(save_path, self.embeddings, allow_dangerous_deserialization=True)

class DataLoader:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path

    def load_data(self):
        data = json.loads(Path(self.json_file_path).read_text(encoding='utf-8'))
        return data
    
    def create_documents(self, length=None):
        data = self.load_data()
        if length is None:
            length = len(data)
        
        documents = [
            Document(
                page_content = self.get_page_content(item),
                metadata = item
            )
            for item in data[:length]
        ]
        return documents
    
    def get_page_content(self, item):
        raise NotImplementedError("Subclasses should implement this method to extract page content from the item")


class BookDataLoader(DataLoader):
    def get_page_content(self, item):
        return f"{item['title']} {item['author']} {item['publication_date']} {item['description']} {' '.join(item['genres'])}"

class MovieDataLoader(DataLoader):
    def get_page_content(self, item):
        return f"{item['title']} {item['release_date']} {item['summary']} {' '.join(item['movie_genres_list'])} {' '.join(item['movie_actor_list'])}"
    
def process_data(json_file_path, model_name, save_path, data_loader_class, length=None):
    # Initialize the embedding manager with the chosen model
    embedding_manager = EmbeddingManager(model_name)

    # Initialize the vector space manager with the embedding manager
    vector_space_manager = VectorSpaceManager(embedding_manager)

    # Load data and create documents
    data_loader = data_loader_class(json_file_path)
    documents = data_loader.create_documents(length)

    # Create and save the vector space
    vector_store = vector_space_manager.create_vector_store(documents)
    vector_space_manager.save_vector_space(vector_store, save_path)

    # Load the vector space and perform a search
    vector_store = vector_space_manager.load_vector_space(save_path)
    query = "The Hobbit"
    search_results = vector_store.search(query, k=2, search_type="similarity")
    print(search_results)


if __name__ == '__main__':
    #book
    json_file_path = 'data/BookSummaries/book.json'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    save_path = 'data/book_vector_store'
    process_data(json_file_path, model_name, save_path, BookDataLoader, 100)

    #movie
    json_file_path = 'data/MovieSummaries/movie.json'
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    save_path = 'data/movie_vector_store'
    process_data(json_file_path, model_name, save_path, MovieDataLoader, 100)