import math

import neo4j
import random
from agents.conversation_summarizer_agent import get_summarizer
from chat_neo4j_etl.ingest_basic_schema import get_distinct_days


def chunk_messages(messages, chunk_size, overlap_size):
    chunks = []
    for i in range(0, len(messages), overlap_size):
        chunk = messages[i:i + chunk_size] if i + chunk_size < len(messages) else messages[i:]
        chunks.append(chunk)

    return chunks


def convert_chunk_to_string(chunk):
    chunk_str = ""

    for message in chunk:
        chunk_str += f"{message['sender']}: {message['message']}\n"

    return chunk_str


def get_summary_info(response):
    response = response.content
    response_lines = response.split("\n")
    # print(response_lines)
    summary_line = response_lines[1]
    sentiment_line = response_lines[2]
    summary = summary_line.split("\"Summary\": ")[1]
    sentiment = sentiment_line.split("\"Sentiment\": ")[1]
    result = {"Summary": summary.replace("\"", ""), "Sentiment": sentiment.replace("\"", "")}

    return result


def random_sample(message_chunks, k):
    return random.sample(message_chunks, k)


def summarize_conversations():
    summary_chain = get_summarizer()
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    query = """
        MATCH (u:User)-[r]->(m:Message)
        WHERE r.day = $day AND r.month = $month AND r.year = $year
        RETURN u.name as sender, m.text as message, id(u) as user_id, id(m) as message_id
        ORDER BY r.hours, r.minutes
        """

    dates = get_distinct_days()
    chunk_ids = 0
    chunk_data = {}

    dates = random_sample(dates, 100)
    date_ctr = 0
    for date in dates:
        messages = []
        print(f"Processing date {date_ctr}/{len(dates)}: {date}")
        date_ctr += 1
        with driver.session() as session:
            res = session.run(query, day=int(date["day"]), month=int(date["month"]), year=int(date["year"]))

            for row in res:
                messages.append({"sender": row["sender"], "message": row["message"], "user_id": row["user_id"],
                                 "message_id": row["message_id"]})

        # Write a function to chunk messages such that each chunk has 30 messages and there is a 10 message overlap between chunks
        message_chunks = chunk_messages(messages, 30, 15)
        # Write a function to pick k chunks from message_chunks
        message_chunks = random_sample(message_chunks, min(5, len(message_chunks)))
        for chunk in message_chunks:
            print("Processing chunk: ", chunk_ids)
            message_neo4j_ids = [message["message_id"] for message in chunk]
            chunk_data[chunk_ids] = {"message_neo4j_ids": message_neo4j_ids}
            chunk_str = convert_chunk_to_string(chunk)
            response = summary_chain.invoke({"conversation": chunk_str})
            try:
                summary_info = get_summary_info(response)
            except Exception as e:
                continue
            chunk_data[chunk_ids]["summary_info"] = summary_info
            chunk_data[chunk_ids]["date"] = date
            chunk_ids += 1

    return chunk_data


def create_golden_dataset():
    chunk_data = summarize_conversations()

    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    query = """
        WITH $chunk_data as chunk_data
        CREATE (c:Conversation)
        SET c.summary= chunk_data.summary_info.Summary, c.sentiment= chunk_data.summary_info.Sentiment
        WITH c, chunk_data
        MATCH (m:Message)
        WHERE id(m) in chunk_data.message_neo4j_ids
        MERGE (c)-[:MESSAGES_IN_CONVERSATION]->(m)
    """
    for id, data in chunk_data.items():
        with driver.session() as session:
            session.run(query, chunk_data=data)


if __name__ == "__main__":
    create_golden_dataset()
