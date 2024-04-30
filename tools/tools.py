import neo4j
from langchain_community.embeddings import HuggingFaceEmbeddings


def query_vector_db_in_range(question, day, month, year):
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    embedding_generator = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    question_embedding = embedding_generator.encode(question)

    query = """
    WITH $question_embedding as question_embedding
    CALL db.index.vector.queryNodes('conversation_text_index', 10, question_embedding) YIELD node as node, score as score
    MATCH (node)-[r1:MESSAGES_IN_CONVERSATION]->(m:Message)-[r2:SENT]->()
    WHERE r2.day = $day AND r2.month = $month AND r2.year = $year
    
    """
