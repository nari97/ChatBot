import neo4j
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

from agents.citation_verification_agent import get_agent
from agents.query_retriever_agent import get_retriever_chain
from agents.timeframe_detection_agent import get_timeframe_agent


def format_context(context):
    return "\n\n".join(
        [f"Summary: {c.page_content}\nDate sent: {c.metadata.day}/{c.metadata.month}/{c.metadata.year}" for c in
         context])


def format_claim_with_context(question, claim, context):
    return_str = ""
    return_str += f"Question: {question}\nClaim: {claim}\nContext:\n"
    for c in context:
        return_str += f"Source ID: {c.metadata['conversation_id']}\nSource summary: {c.page_content}\nSource date: Date sent: {c.metadata.day}/{c.metadata.month}/{c.metadata.year}\n\n"

    return return_str


def parse_timeframe_response(content):
    lines = content.split("\n")
    question = lines[0].split(": ")[1].strip()
    timeframe_start = lines[1].split(":")[1].strip()
    timeframe_end = lines[2].split(":")[1].strip()

    timeframe_start = [int(val) for val in timeframe_start.split("-")]
    timeframe_end = [int(val) for val in timeframe_end.split("-")]

    return question, timeframe_start, timeframe_end


def answer_question(question, vector_index):
    timeframe_chain = get_timeframe_agent()
    response = timeframe_chain.invoke({"question": question})

    question, timeframe_start, timeframe_end = parse_timeframe_response(response.content)
    print("Question: ", question)
    print("Timeframe start: ", timeframe_start)
    print("Timeframe end: ", timeframe_end)

    context = vector_index.similarity_search(question, k=10)
    context_str = format_context(context)

    retriever_chain = get_retriever_chain()
    response = retriever_chain.invoke({"context": context_str, "question": question})

    claim = response.content

    citation_chain = get_agent()
    citation_context = format_claim_with_context(question, claim, context)

    citation_response = citation_chain.invoke({"context": citation_context})

    print(citation_context)

    print("Edited claim: ", citation_response.content.split("\n")[0])


def get_faiss_vectorstore():
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    query = """
    MATCH (c:Conversation) 
    RETURN id(c) as conversation_id, c.summary as summary, c.day as day, c.month as month, c.year as year
    """

    documents = []

    with driver.session() as session:
        res = session.run(query)
        for result in res:
            documents.append(
                Document(page_content=result["summary"],
                         metadata={"conversation_id": result["conversation_id"], "day": result["day"],
                                   "month": result["month"], "year": result["year"]}))

    faiss_vector_index = FAISS.from_documents(documents,
                                              HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                                                    encode_kwargs={
                                                                        "normalize_embeddings": True}))

    return faiss_vector_index


if __name__ == "__main__":
    question = "What does Nikhil think about haaland since the start of the year?"

    faiss_vector_index = get_faiss_vectorstore()
    answer_question(question, faiss_vector_index)
