import re
import neo4j


def get_date_json(date):
    date_splits = date.split("/")
    return {
        "day": int(date_splits[1]),
        "month": int(date_splits[0]),
        "year": int("20" + date_splits[2])
    }


def get_time_json(time):
    hours = int(time[0:2]) if ":" not in time else int(time[0])
    minutes = int(time[3:5])
    time_of_day = time[5:]

    if time_of_day == "PM":
        hours += 12

    return {
        "hours": hours,
        "minutes": minutes,
    }


def remove_emojis(data):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def get_chat_as_lines():
    with open("../data/chats.txt", encoding="utf-8") as f:
        return f.readlines()


def parse_lines_to_json(lines):
    chats = []

    for line in lines:
        # print(line)
        line = line.strip()
        splits = line.split(",")
        date = splits[0]
        rest = ",".join(splits[1:])

        rest_splits = rest.split("-")
        time = rest_splits[0].strip()
        time = time[0:5] + time[-2:]
        rest = "-".join(rest_splits[1:])

        rest_splits = rest.split(":")
        user = rest_splits[0].strip()
        message = ":".join(rest_splits[1:]).strip()

        if message == "<Media omitted>" or "x.com" in message:
            continue

        if user == "":
            message = remove_emojis(date)
            if message == "":
                continue

            date = chats[-1]["date"]
            time = chats[-1]["time"]
            user = chats[-1]["user"]
            user = user.replace("+91 88867 02075", "Tushar")
            message = message.replace("+91 88867 02075", "Tushar")
            chats.append({
                "date": date,
                "time": time,
                "user": user,
                "message": message.lower()
            })
        else:
            message = remove_emojis(message)
            if message == "":
                continue
            user = user.replace("+91 88867 02075", "Tushar")
            message = message.replace("+91 88867 02075", "Tushar")
            chats.append({
                "date": get_date_json(date),
                "time": get_time_json(time),
                "user": user,
                "message": message.lower()
            })

    return chats


def load_data_to_neo4j(chats):
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    query = """
    UNWIND $chats AS chat
    MERGE (u:User {name: chat.user})
    CREATE (m:Message {text: chat.message})
    MERGE (u)-[r:SENT]->(m)
    SET r.day = chat.date.day, r.month = chat.date.month, r.year = chat.date.year, r.hours = chat.time.hours, r.minutes = chat.time.minutes
    """

    with driver.session() as session:
        res = session.run(query, chats=chats)
        for result in res:
            print(result)


def get_distinct_days():
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    query = """
        MATCH ()-[r]->()
        RETURN DISTINCT r.day as day, r.month as month, r.year as year
        """

    dates = []
    with driver.session() as session:
        res = session.run(query)
        for result in res:
            dates.append({"day": result["day"], "month": result["month"], "year": result["year"]})

    return dates


def get_message_ids_for_day_ordered(day, month, year):
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    query = """
        MATCH ()-[r]->(m:Message)
        WHERE r.day = $day AND r.month = $month AND r.year = $year
        RETURN id(m) as id
        ORDER BY r.hours, r.minutes
        """

    message_ids = []
    with driver.session() as session:
        res = session.run(query, day=day, month=month, year=year)
        for result in res:
            message_ids.append(result["id"])

    return message_ids


def connect_messages(message_pairs):
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    query = """
        UNWIND $pairs AS pair
        MATCH (m1:Message), (m2:Message)
        WHERE id(m1) = pair[0] AND id(m2) = pair[1]
        MERGE (m1)-[:NEXT_MESSAGE_IN_DAY]->(m2)
        """

    with driver.session() as session:
        session.run(query, pairs=message_pairs)


def create_conversation_by_days():
    unique_days = get_distinct_days()

    day_ctr = 1
    for day in unique_days:
        print(f"Retrieving and connecting messages for day {day_ctr} of {len(unique_days)}: {day}")
        day_ctr += 1
        message_pairs = []
        message_ids = get_message_ids_for_day_ordered(day["day"], day["month"], day["year"])

        for i in range(0, len(message_ids) - 1):
            message_pairs.append((message_ids[i], message_ids[i + 1]))

        connect_messages(message_pairs)


def load_chats():
    k = 10000

    print(f"Loading last {k} chats")
    chat_lines = get_chat_as_lines()

    # Load last k lines in chat_lines
    parsed_chats = parse_lines_to_json(chat_lines[-k:])

    load_data_to_neo4j(parsed_chats)
    print(f"Loaded {len(parsed_chats)} chats")


def clear_database():
    print("Clearing database")
    driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    query = """
    MATCH (n)
    DETACH DELETE n
    """

    with driver.session() as session:
        session.run(query)

    print("Database cleared")


if __name__ == "__main__":
    clear_database()
    load_chats()
    # create_conversation_by_days()
