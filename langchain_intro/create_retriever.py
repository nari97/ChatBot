from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

loader = CSVLoader(REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

reviews_vector_db = Chroma.from_documents(reviews, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                                          persist_directory=REVIEWS_CHROMA_PATH)
