import neo4j

from agents.entity_recognition_agent import get_er_agent


def get_entities(response):
    print(response.content)


def recognize_entities():
    er_agent = get_er_agent()
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    query = "MATCH (c:Conversation) WITH c, rand() as r RETURN id(c) as id, c.summary as summary ORDER BY r"

    data = {}
    with driver.session() as session:
        res = session.run(query)

        for row in res:
            conversation_id = row["id"]
            summary = row["summary"]
            response = er_agent.invoke({"summary": summary})
            print(summary)

            people_entities, organization_entities = get_entities(response)

            exit()


if __name__ == "__main__":
    recognize_entities()
