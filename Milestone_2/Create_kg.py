import pandas as pd
from neo4j import GraphDatabase

def load_config():
    cfg = {}
    with open("config.txt", "r") as f:
        for line in f:
            key, val = line.strip().split("=")
            cfg[key] = val
    return cfg["URI"], cfg["USERNAME"], cfg["PASSWORD"]

def create_constraints(session):
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE")

def create_nodes(tx, users, hotels, reviews, hotel_avg_score):
    tx.run("MATCH (n) DETACH DELETE n")

    for _, row in users.iterrows():
        # Traveller Nodes
        tx.run("""
            MERGE (t:Traveller {user_id: $user_id})
            SET t.age = $age,
                t.type = $type,
                t.gender = $gender
        """, {
            "user_id": row["user_id"],
            "age": row["age_group"],
            "type": row["traveller_type"],
            "gender": row["user_gender"]
        })

        # From country relation
        tx.run("""
            MERGE (c:Country {name: $country})
            WITH c
            MATCH (t:Traveller {user_id: $user_id})
            MERGE (t)-[:FROM_COUNTRY]->(c)
        """, {
            "user_id": row["user_id"],
            "country": row["country"]
        })


    for _, row in hotels.iterrows():

        # Hotel Nodes
        tx.run("""
            MERGE (h:Hotel {hotel_id: $hotel_id})
            SET h.name = $name,
                h.star_rating = $star_rating,
                h.cleanliness_base = $cleanliness_base,
                h.comfort_base = $comfort_base,
                h.facilities_base = $facilities_base,
                h.average_reviews_score = $average_reviews_score
        """, {
            "hotel_id": row["hotel_id"],
            "name": row["hotel_name"],
            "star_rating": row["star_rating"],
            "cleanliness_base": row["cleanliness_base"],
            "comfort_base": row["comfort_base"],
            "facilities_base": row["facilities_base"],
            "average_reviews_score": hotel_avg_score.get(row["hotel_id"], 0) 
        })

        # City Country relation
        tx.run("""
            MERGE (city:City {name: $city})
            MERGE (country:Country {name: $country})
            MERGE (city)-[:LOCATED_IN]->(country)
        """, {
            "city": row["city"],
            "country": row["country"]
        })

        # Hotel City relation
        tx.run("""
            MATCH (h:Hotel {hotel_id: $hotel_id})
            MATCH (city:City {name: $city})
            MERGE (h)-[:LOCATED_IN]->(city)
        """, {
            "hotel_id": row["hotel_id"],
            "city": row["city"]
        })


    for _, row in reviews.iterrows():

        # review nodes
        tx.run("""
            MERGE (r:Review {review_id: $review_id})
            SET r.text = $text,
                r.date = $date,
                r.score_overall = $score_overall,
                r.score_cleanliness = $score_cleanliness,
                r.score_comfort = $score_comfort,
                r.score_facilities = $score_facilities,
                r.score_location = $score_location,
                r.score_staff = $score_staff,
                r.score_value_for_money = $score_value_for_money
        """, {
            "review_id": row["review_id"],
            "text": row["review_text"],
            "date": row["review_date"],
            "score_overall": row["score_overall"],
            "score_cleanliness": row["score_cleanliness"],
            "score_comfort": row["score_comfort"],
            "score_facilities": row["score_facilities"],
            "score_location": row["score_location"],
            "score_staff": row["score_staff"],
            "score_value_for_money": row["score_value_for_money"]
        })

def create_relationships(tx, reviews, visa):

    for _, row in reviews.iterrows():
        tx.run("""
            MATCH (t:Traveller {user_id: $user_id})
            MATCH (h:Hotel {hotel_id: $hotel_id})
            MATCH (r:Review {review_id: $review_id})

            MERGE (t)-[:WROTE]->(r)
            MERGE (r)-[:REVIEWED]->(h)
            MERGE (t)-[:STAYED_AT]->(h)
        """, {
            "user_id": row["user_id"],
            "hotel_id": row["hotel_id"],
            "review_id": row["review_id"]
        })

    for _, row in visa.iterrows():
        requires_visa = row["requires_visa"]

        if str(requires_visa).strip().lower() == "yes":
            tx.run("""
                MERGE (c1:Country {name: $from_c})
                MERGE (c2:Country {name: $to_c})
                MERGE (c1)-[:NEEDS_VISA {visa_type: $visa_type}]->(c2)
            """, {
                "from_c": row["from"],
                "to_c": row["to"],
                "visa_type": row["visa_type"]
            })


def main():
    uri, user, password = load_config()
    driver = GraphDatabase.driver(uri, auth=(user, password))

    users = pd.read_csv("users.csv")
    hotels = pd.read_csv("hotels.csv")
    reviews = pd.read_csv("reviews.csv")
    visa = pd.read_csv("visa.csv")

    hotel_avg_score = reviews.groupby("hotel_id")["score_overall"].mean().to_dict()

    with driver.session() as session:
        print("Creating constraints")
        create_constraints(session)

        print("Creating nodes")
        create_nodes(session, users, hotels, reviews, hotel_avg_score)

        print("Creating relationships")
        create_relationships(session, reviews, visa)

        print("Knowledge Graph creation completed successfully!")


if __name__ == "__main__":
    main()