# -*- coding: utf-8 -*-
"""
Baseline Cypher Queries for Graph-RAG Hotel Assistant
Milestone 3 - Advanced Computational Linguistics

This module contains 20 Cypher query templates:
- 10 for Booking and Visa Assistant
- 10 for Hotel Recommender System
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class HotelCypherQueries:
    """
    Baseline Cypher query templates for hotel knowledge graph.
    Each query is parameterized and can be filled with extracted entities.
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results as list of dictionaries."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    # ========================================================================
    # BOOKING AND VISA ASSISTANT QUERIES (10 queries)
    # ========================================================================
    
    def query_1_hotels_in_city(self, city: str, min_rating: float = 0.0) -> List[Dict]:
        """
        Query 1: Find hotels in a specific city with minimum rating
        Intent: Location-based hotel search
        Entities: city, min_rating (optional)
        Example: "Find hotels in Paris", "Hotels in Cairo with rating > 8"
        """
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City {name: $city})
        WHERE h.average_reviews_score >= $min_rating
        RETURN h.name as hotel_name, 
               h.star_rating as stars,
               h.average_reviews_score as rating,
               city.name as city,
               h.cleanliness_base as cleanliness,
               h.comfort_base as comfort,
               h.facilities_base as facilities
        ORDER BY h.average_reviews_score DESC
        """
        return self.execute_query(query, {"city": city, "min_rating": min_rating})
    
    def query_2_hotels_in_country(self, country: str, min_stars: int = 0) -> List[Dict]:
        """
        Query 2: Find hotels in a specific country with minimum star rating
        Intent: Country-based hotel search
        Entities: country, min_stars (optional)
        Example: "Hotels in France", "5-star hotels in Egypt"
        """
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country {name: $country})
        WHERE h.star_rating >= $min_stars
        RETURN h.name as hotel_name,
               city.name as city,
               country.name as country,
               h.star_rating as stars,
               h.average_reviews_score as rating
        ORDER BY h.average_reviews_score DESC, h.star_rating DESC
        """
        return self.execute_query(query, {"country": country, "min_stars": min_stars})
    
    def query_3_visa_requirements(self, from_country: str, to_country: str) -> List[Dict]:
        """
        Query 3: Check visa requirements between countries
        Intent: Visa information lookup
        Entities: from_country, to_country
        Example: "Do I need a visa from USA to France?", "Visa requirements Egypt to UK"
        """
        query = """
        MATCH (from:Country {name: $from_country})
        OPTIONAL MATCH (from)-[v:NEEDS_VISA]->(to:Country {name: $to_country})
        RETURN from.name as from_country,
               to.name as to_country,
               CASE WHEN v IS NOT NULL THEN 'Yes' ELSE 'No' END as requires_visa,
               v.visa_type as visa_type
        """
        return self.execute_query(query, {"from_country": from_country, "to_country": to_country})
    
    def query_4_hotels_by_amenity_score(self, city: str, amenity: str, min_score: float = 8.0) -> List[Dict]:
        """
        Query 4: Find hotels by specific amenity/feature score
        Intent: Amenity-based search
        Entities: city, amenity (cleanliness/comfort/facilities), min_score
        Example: "Hotels in Tokyo with good cleanliness", "Hotels in Dubai with best facilities"
        """
        # Map amenity name to property
        amenity_map = {
            "cleanliness": "cleanliness_base",
            "comfort": "comfort_base",
            "facilities": "facilities_base",
            "location": "location_base",
            "staff": "staff_base",
            "value": "value_for_money_base"
        }
        
        amenity_prop = amenity_map.get(amenity.lower(), "facilities_base")
        
        query = f"""
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City {{name: $city}})
        WHERE h.{amenity_prop} >= $min_score
        RETURN h.name as hotel_name,
               h.{amenity_prop} as {amenity}_score,
               h.average_reviews_score as overall_rating,
               city.name as city
        ORDER BY h.{amenity_prop} DESC
        LIMIT 10
        """
        return self.execute_query(query, {"city": city, "min_score": min_score})
    
    def query_5_hotel_reviews(self, hotel_name: str, min_score: float = 0.0) -> List[Dict]:
        """
        Query 5: Get reviews for a specific hotel
        Intent: Review lookup
        Entities: hotel_name, min_score (optional)
        Example: "Show me reviews for The Azure Tower", "Good reviews for Nile Grandeur"
        """
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)
        WHERE r.score_overall >= $min_score
        RETURN r.text as review_text,
               r.score_overall as overall_score,
               r.score_cleanliness as cleanliness,
               r.score_comfort as comfort,
               r.score_facilities as facilities,
               r.score_location as location,
               r.score_staff as staff,
               r.date as review_date
        ORDER BY r.score_overall DESC, r.date DESC
        LIMIT 20
        """
        return self.execute_query(query, {"hotel_name": hotel_name, "min_score": min_score})
    
    def query_6_hotels_by_traveller_type(self, city: str, traveller_type: str) -> List[Dict]:
        """
        Query 6: Find hotels popular with specific traveller type
        Intent: Traveller-type based recommendation
        Entities: city, traveller_type (Solo, Couple, Family, Business, Group)
        Example: "Hotels in Paris for families", "Best hotels for solo travelers in Tokyo"
        """
        query = """
        MATCH (t:Traveller {type: $traveller_type})-[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(city:City {name: $city})
        WITH h, city, COUNT(t) as visitor_count
        RETURN h.name as hotel_name,
               city.name as city,
               h.average_reviews_score as rating,
               visitor_count as popularity,
               h.star_rating as stars
        ORDER BY visitor_count DESC, h.average_reviews_score DESC
        LIMIT 10
        """
        return self.execute_query(query, {"city": city, "traveller_type": traveller_type})
    
    def query_7_hotel_details(self, hotel_name: str) -> List[Dict]:
        """
        Query 7: Get complete details for a specific hotel
        Intent: Hotel information lookup
        Entities: hotel_name
        Example: "Tell me about The Royal Compass", "Details for Marina Bay Zenith"
        """
        query = """
        MATCH (h:Hotel {name: $hotel_name})-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WITH h, city, country, 
             COUNT(r) as review_count,
             AVG(r.score_overall) as avg_score
        RETURN h.name as hotel_name,
               city.name as city,
               country.name as country,
               h.star_rating as stars,
               h.average_reviews_score as rating,
               review_count,
               h.cleanliness_base as cleanliness,
               h.comfort_base as comfort,
               h.facilities_base as facilities,
               h.location_base as location_score,
               h.staff_base as staff,
               COALESCE(h.value_for_money_base, 0.0) as value_for_money
        """
        return self.execute_query(query, {"hotel_name": hotel_name})
    
    def query_8_compare_hotels(self, hotel_names: List[str]) -> List[Dict]:
        """
        Query 8: Compare multiple hotels side by side
        Intent: Hotel comparison
        Entities: hotel_names (list)
        Example: "Compare The Azure Tower and The Royal Compass"
        """
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)
        WHERE h.name IN $hotel_names
        RETURN h.name as hotel_name,
               city.name as city,
               h.star_rating as stars,
               h.average_reviews_score as rating,
               h.cleanliness_base as cleanliness,
               h.comfort_base as comfort,
               h.facilities_base as facilities,
               COALESCE(h.value_for_money_base, 0.0) as value
        ORDER BY h.average_reviews_score DESC
        """
        return self.execute_query(query, {"hotel_names": hotel_names})
    
    def query_9_countries_requiring_visa(self, from_country: str) -> List[Dict]:
        """
        Query 9: Find all countries requiring visa from a specific country
        Intent: Visa planning
        Entities: from_country
        Example: "Which countries need visa from USA?", "Visa requirements from Egypt"
        """
        query = """
        MATCH (from:Country {name: $from_country})-[v:NEEDS_VISA]->(to:Country)
        RETURN to.name as destination_country,
               v.visa_type as visa_type
        ORDER BY to.name
        """
        return self.execute_query(query, {"from_country": from_country})
    
    def query_10_top_rated_hotels_globally(self, min_rating: float = 9.0, limit: int = 10) -> List[Dict]:
        """
        Query 10: Find top-rated hotels globally
        Intent: Best hotels discovery
        Entities: min_rating (optional), limit (optional)
        Example: "Show me the best hotels", "Top 5 hotels worldwide"
        """
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
        WHERE h.average_reviews_score >= $min_rating
        RETURN h.name as hotel_name,
               city.name as city,
               country.name as country,
               h.star_rating as stars,
               h.average_reviews_score as rating,
               h.cleanliness_base as cleanliness,
               h.comfort_base as comfort,
               h.facilities_base as facilities
        ORDER BY h.average_reviews_score DESC, h.star_rating DESC
        LIMIT $limit
        """
        return self.execute_query(query, {"min_rating": min_rating, "limit": limit})
    
    # ========================================================================
    # HOTEL RECOMMENDER SYSTEM QUERIES (10 queries)
    # ========================================================================
    
    def query_11_similar_hotels_by_location(self, hotel_name: str, limit: int = 5) -> List[Dict]:
        """
        Query 11: Find hotels in the same city as a given hotel
        Intent: Location-based recommendations
        Entities: hotel_name, limit (optional)
        Example: "Hotels similar to The Azure Tower in the same city"
        """
        query = """
        MATCH (h1:Hotel {name: $hotel_name})-[:LOCATED_IN]->(city:City)<-[:LOCATED_IN]-(h2:Hotel)
        WHERE h1 <> h2
        RETURN h2.name as hotel_name,
               city.name as city,
               h2.star_rating as stars,
               h2.average_reviews_score as rating,
               ABS(h1.average_reviews_score - h2.average_reviews_score) as rating_difference
        ORDER BY rating_difference ASC, h2.average_reviews_score DESC
        LIMIT $limit
        """
        return self.execute_query(query, {"hotel_name": hotel_name, "limit": limit})
    
    def query_12_hotels_liked_by_similar_travelers(self, hotel_name: str, limit: int = 5) -> List[Dict]:
        """
        Query 12: Collaborative filtering - hotels liked by users who liked this hotel
        Intent: User-based collaborative filtering
        Entities: hotel_name, limit (optional)
        Example: "Recommend hotels based on users who liked The Royal Compass"
        """
        query = """
        MATCH (h1:Hotel {name: $hotel_name})<-[:STAYED_AT]-(t:Traveller)-[:STAYED_AT]->(h2:Hotel)
        WHERE h1 <> h2
        WITH h2, COUNT(DISTINCT t) as shared_travelers
        MATCH (h2)-[:LOCATED_IN]->(city:City)
        RETURN h2.name as hotel_name,
               city.name as city,
               h2.average_reviews_score as rating,
               shared_travelers as popularity
        ORDER BY shared_travelers DESC, h2.average_reviews_score DESC
        LIMIT $limit
        """
        return self.execute_query(query, {"hotel_name": hotel_name, "limit": limit})
    
    def query_13_hotels_by_similar_ratings(self, hotel_name: str, tolerance: float = 0.5, limit: int = 5) -> List[Dict]:
        """
        Query 13: Find hotels with similar overall ratings
        Intent: Rating-based similarity
        Entities: hotel_name, tolerance (optional), limit (optional)
        Example: "Hotels with similar quality to Marina Bay Zenith"
        """
        query = """
        MATCH (h1:Hotel {name: $hotel_name})
        MATCH (h2:Hotel)-[:LOCATED_IN]->(city:City)
        WHERE h1 <> h2 
          AND ABS(h1.average_reviews_score - h2.average_reviews_score) <= $tolerance
        RETURN h2.name as hotel_name,
               city.name as city,
               h2.average_reviews_score as rating,
               h2.star_rating as stars,
               ABS(h1.average_reviews_score - h2.average_reviews_score) as rating_difference
        ORDER BY rating_difference ASC
        LIMIT $limit
        """
        return self.execute_query(query, {"hotel_name": hotel_name, "tolerance": tolerance, "limit": limit})
    
    def query_14_hotels_for_traveller_profile(self, age_group: str, traveller_type: str, 
                                               country: str = None, limit: int = 5) -> List[Dict]:
        """
        Query 14: Recommend hotels based on traveller demographics
        Intent: Demographic-based recommendations
        Entities: age_group, traveller_type, country (optional), limit (optional)
        Example: "Hotels for young solo travelers", "Best hotels for senior couples in France"
        """
        if country:
            query = """
            MATCH (t:Traveller {age: $age_group, type: $traveller_type})-[s:STAYED_AT]->(h:Hotel)
                  -[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country {name: $country})
            WITH h, city, COUNT(s) as visits
            RETURN h.name as hotel_name,
                   city.name as city,
                   h.average_reviews_score as rating,
                   visits as popularity
            ORDER BY visits DESC, h.average_reviews_score DESC
            LIMIT $limit
            """
            params = {"age_group": age_group, "traveller_type": traveller_type, 
                     "country": country, "limit": limit}
        else:
            query = """
            MATCH (t:Traveller {age: $age_group, type: $traveller_type})-[s:STAYED_AT]->(h:Hotel)
                  -[:LOCATED_IN]->(city:City)
            WITH h, city, COUNT(s) as visits
            RETURN h.name as hotel_name,
                   city.name as city,
                   h.average_reviews_score as rating,
                   visits as popularity
            ORDER BY visits DESC, h.average_reviews_score DESC
            LIMIT $limit
            """
            params = {"age_group": age_group, "traveller_type": traveller_type, "limit": limit}
        
        return self.execute_query(query, params)
    
    def query_15_hotels_with_best_amenity(self, amenity: str, country: str = None, limit: int = 5) -> List[Dict]:
        """
        Query 15: Recommend hotels with highest score in specific amenity
        Intent: Amenity-focused recommendations
        Entities: amenity, country (optional), limit (optional)
        Example: "Hotels with best facilities in Japan", "Cleanest hotels worldwide"
        """
        amenity_map = {
            "cleanliness": "cleanliness_base",
            "comfort": "comfort_base",
            "facilities": "facilities_base",
            "location": "location_base",
            "staff": "staff_base",
            "value": "value_for_money_base"
        }
        
        amenity_prop = amenity_map.get(amenity.lower(), "facilities_base")
        
        if country:
            query = f"""
            MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country {{name: $country}})
            RETURN h.name as hotel_name,
                   city.name as city,
                   h.{amenity_prop} as {amenity}_score,
                   h.average_reviews_score as overall_rating
            ORDER BY h.{amenity_prop} DESC
            LIMIT $limit
            """
            params = {"country": country, "limit": limit}
        else:
            query = f"""
            MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)
            RETURN h.name as hotel_name,
                   city.name as city,
                   h.{amenity_prop} as {amenity}_score,
                   h.average_reviews_score as overall_rating
            ORDER BY h.{amenity_prop} DESC
            LIMIT $limit
            """
            params = {"limit": limit}
        
        return self.execute_query(query, params)
    
    def query_16_hotels_in_same_country(self, hotel_name: str, limit: int = 5) -> List[Dict]:
        """
        Query 16: Find other highly-rated hotels in the same country
        Intent: Country-based recommendations
        Entities: hotel_name, limit (optional)
        Example: "Other great hotels in the same country as L'Étoile Palace"
        """
        query = """
        MATCH (h1:Hotel {name: $hotel_name})-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(country:Country)
              <-[:LOCATED_IN]-(:City)<-[:LOCATED_IN]-(h2:Hotel)
        WHERE h1 <> h2
        MATCH (h2)-[:LOCATED_IN]->(city2:City)
        RETURN h2.name as hotel_name,
               city2.name as city,
               country.name as country,
               h2.average_reviews_score as rating,
               h2.star_rating as stars
        ORDER BY h2.average_reviews_score DESC
        LIMIT $limit
        """
        return self.execute_query(query, {"hotel_name": hotel_name, "limit": limit})
    
    def query_17_diverse_recommendations(self, min_rating: float = 8.5, limit: int = 5) -> List[Dict]:
        """
        Query 17: Get diverse hotel recommendations from different countries
        Intent: Diverse/exploratory recommendations
        Entities: min_rating (optional), limit (optional)
        Example: "Show me great hotels from different countries"
        """
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(country:Country)
        WHERE h.average_reviews_score >= $min_rating
        WITH country, h, city
        ORDER BY h.average_reviews_score DESC
        WITH country, COLLECT({hotel: h.name, city: city.name, rating: h.average_reviews_score})[0] as best_hotel
        RETURN best_hotel.hotel as hotel_name,
               best_hotel.city as city,
               country.name as country,
               best_hotel.rating as rating
        ORDER BY best_hotel.rating DESC
        LIMIT $limit
        """
        return self.execute_query(query, {"min_rating": min_rating, "limit": limit})
    
    def query_18_hotels_with_positive_reviews(self, min_review_score: float = 9.0, limit: int = 5) -> List[Dict]:
        """
        Query 18: Find hotels with consistently high review scores
        Intent: Quality-based recommendations
        Entities: min_review_score (optional), limit (optional)
        Example: "Hotels with excellent reviews", "Most highly reviewed hotels"
        """
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)
        WHERE r.score_overall >= $min_review_score
        WITH h, COUNT(r) as high_score_reviews, AVG(r.score_overall) as avg_review_score
        MATCH (h)-[:LOCATED_IN]->(city:City)
        RETURN h.name as hotel_name,
               city.name as city,
               h.average_reviews_score as overall_rating,
               high_score_reviews as excellent_review_count,
               avg_review_score as average_review_score
        ORDER BY high_score_reviews DESC, avg_review_score DESC
        LIMIT $limit
        """
        return self.execute_query(query, {"min_review_score": min_review_score, "limit": limit})
    
    def query_19_budget_friendly_hotels(self, city: str = None, limit: int = 5) -> List[Dict]:
        """
        Query 19: Find hotels with best value for money
        Intent: Budget-conscious recommendations
        Entities: city (optional), limit (optional)
        Example: "Best value hotels in Paris", "Budget-friendly hotels"
        """
        if city:
            query = """
            MATCH (h:Hotel)-[:LOCATED_IN]->(city:City {name: $city})
            RETURN h.name as hotel_name,
                   city.name as city,
                   COALESCE(h.value_for_money_base, 0.0) as value_score,
                   h.average_reviews_score as rating,
                   h.star_rating as stars
            ORDER BY COALESCE(h.value_for_money_base, 0.0) DESC, h.average_reviews_score DESC
            LIMIT $limit
            """
            params = {"city": city, "limit": limit}
        else:
            query = """
            MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)
            RETURN h.name as hotel_name,
                   city.name as city,
                   COALESCE(h.value_for_money_base, 0.0) as value_score,
                   h.average_reviews_score as rating,
                   h.star_rating as stars
            ORDER BY COALESCE(h.value_for_money_base, 0.0) DESC, h.average_reviews_score DESC
            LIMIT $limit
            """
            params = {"limit": limit}
        
        return self.execute_query(query, params)
    
    def query_20_hotels_by_combined_scores(self, priorities: Dict[str, float], 
                                            city: str = None, limit: int = 5) -> List[Dict]:
        """
        Query 20: Personalized recommendations based on weighted preferences
        Intent: Personalized multi-criteria recommendations
        Entities: priorities (dict of amenity:weight), city (optional), limit (optional)
        Example: "Hotels prioritizing cleanliness and comfort in Tokyo"
        
        priorities example: {"cleanliness": 0.4, "comfort": 0.3, "facilities": 0.3}
        """
        # Default equal weights if not provided
        if not priorities:
            priorities = {
                "cleanliness": 0.2,
                "comfort": 0.2,
                "facilities": 0.2,
                "location": 0.2,
                "staff": 0.2
            }
        
        # Build weighted score calculation
        weight_calc = " + ".join([
            f"(h.{amenity}_base * {weight})"
            for amenity, weight in priorities.items()
        ])
        
        if city:
            query = f"""
            MATCH (h:Hotel)-[:LOCATED_IN]->(city:City {{name: $city}})
            WITH h, city, ({weight_calc}) as weighted_score
            RETURN h.name as hotel_name,
                   city.name as city,
                   h.average_reviews_score as rating,
                   weighted_score as personalized_score,
                   h.cleanliness_base as cleanliness,
                   h.comfort_base as comfort,
                   h.facilities_base as facilities
            ORDER BY weighted_score DESC
            LIMIT $limit
            """
            params = {"city": city, "limit": limit}
        else:
            query = f"""
            MATCH (h:Hotel)-[:LOCATED_IN]->(city:City)
            WITH h, city, ({weight_calc}) as weighted_score
            RETURN h.name as hotel_name,
                   city.name as city,
                   h.average_reviews_score as rating,
                   weighted_score as personalized_score,
                   h.cleanliness_base as cleanliness,
                   h.comfort_base as comfort,
                   h.facilities_base as facilities
            ORDER BY weighted_score DESC
            LIMIT $limit
            """
            params = {"limit": limit}
        
        return self.execute_query(query, params)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_queries():
    """Test all 20 query templates with sample data."""
    
    # Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    
    queries = HotelCypherQueries(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    print("="*80)
    print("TESTING BASELINE CYPHER QUERIES")
    print("="*80)
    
    # Test Booking & Visa Assistant Queries
    print("\n" + "="*80)
    print("BOOKING AND VISA ASSISTANT QUERIES (1-10)")
    print("="*80)
    
    print("\n1. Hotels in Paris:")
    results = queries.query_1_hotels_in_city("Paris", min_rating=8.0)
    for r in results[:3]:
        print(f"   - {r['hotel_name']} ({r['stars']}★) - Rating: {r['rating']:.2f}")
    
    print("\n2. Hotels in France:")
    results = queries.query_2_hotels_in_country("France", min_stars=5)
    for r in results[:3]:
        print(f"   - {r['hotel_name']} in {r['city']} - {r['stars']}★")
    
    print("\n3. Visa requirements (USA to France):")
    results = queries.query_3_visa_requirements("United States", "France")
    for r in results:
        print(f"   - Requires visa: {r['requires_visa']}, Type: {r.get('visa_type', 'N/A')}")
    
    print("\n4. Hotels in Tokyo with good facilities:")
    results = queries.query_4_hotels_by_amenity_score("Tokyo", "facilities", min_score=9.0)
    for r in results[:3]:
        print(f"   - {r['hotel_name']} - Facilities: {r['facilities_score']:.1f}")
    
    print("\n5. Reviews for Nile Grandeur:")
    results = queries.query_5_hotel_reviews("Nile Grandeur", min_score=8.0)
    for r in results[:2]:
        print(f"   - Score: {r['overall_score']:.1f} - {r['review_text'][:60]}...")
    
    print("\n6. Hotels in Paris for families:")
    results = queries.query_6_hotels_by_traveller_type("Paris", "Family")
    for r in results[:3]:
        print(f"   - {r['hotel_name']} - Popularity: {r['popularity']} families")
    
    print("\n7. Details for The Royal Compass:")
    results = queries.query_7_hotel_details("The Royal Compass")
    if results:
        r = results[0]
        print(f"   - {r['hotel_name']} in {r['city']}, {r['country']}")
        print(f"   - {r['stars']}★, Rating: {r['rating']:.2f}, Reviews: {r['review_count']}")
    
    print("\n8. Compare hotels:")
    results = queries.query_8_compare_hotels(["The Azure Tower", "The Royal Compass"])
    for r in results:
        print(f"   - {r['hotel_name']}: {r['rating']:.2f} rating, {r['stars']}★")
    
    print("\n9. Countries requiring visa from USA:")
    results = queries.query_9_countries_requiring_visa("United States")
    for r in results[:5]:
        print(f"   - {r['destination_country']} ({r['visa_type']})")
    
    print("\n10. Top rated hotels globally:")
    results = queries.query_10_top_rated_hotels_globally(min_rating=9.0, limit=5)
    for r in results:
        print(f"   - {r['hotel_name']} ({r['city']}, {r['country']}) - {r['rating']:.2f}")
    
    # Test Recommender System Queries
    print("\n" + "="*80)
    print("HOTEL RECOMMENDER SYSTEM QUERIES (11-20)")
    print("="*80)
    
    print("\n11. Similar hotels by location (same city as The Azure Tower):")
    results = queries.query_11_similar_hotels_by_location("The Azure Tower", limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} - Rating: {r['rating']:.2f}")
    
    print("\n12. Collaborative filtering (users who liked The Royal Compass):")
    results = queries.query_12_hotels_liked_by_similar_travelers("The Royal Compass", limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} ({r['city']}) - {r['popularity']} shared travelers")
    
    print("\n13. Hotels with similar ratings to Marina Bay Zenith:")
    results = queries.query_13_hotels_by_similar_ratings("Marina Bay Zenith", tolerance=0.3, limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} - Rating: {r['rating']:.2f}")
    
    print("\n14. Hotels for young solo travelers:")
    results = queries.query_14_hotels_for_traveller_profile("18-24", "Solo", limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} ({r['city']}) - Popularity: {r['popularity']}")
    
    print("\n15. Hotels with best facilities in Japan:")
    results = queries.query_15_hotels_with_best_amenity("facilities", country="Japan", limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} - Facilities: {r['facilities_score']:.1f}")
    
    print("\n16. Other hotels in same country as L'Étoile Palace:")
    results = queries.query_16_hotels_in_same_country("L'Étoile Palace", limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} ({r['city']}) - {r['rating']:.2f}")
    
    print("\n17. Diverse recommendations from different countries:")
    results = queries.query_17_diverse_recommendations(min_rating=9.0, limit=5)
    for r in results:
        print(f"   - {r['hotel_name']} ({r['city']}, {r['country']}) - {r['rating']:.2f}")
    
    print("\n18. Hotels with excellent reviews:")
    results = queries.query_18_hotels_with_positive_reviews(min_review_score=9.0, limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} - {r['excellent_review_count']} excellent reviews")
    
    print("\n19. Best value hotels in Paris:")
    results = queries.query_19_budget_friendly_hotels(city="Paris", limit=3)
    for r in results:
        value = r.get('value_score', 0.0) or 0.0
        print(f"   - {r['hotel_name']} - Value score: {value:.1f}")
    
    print("\n20. Personalized recommendations (prioritizing cleanliness & comfort):")
    priorities = {"cleanliness": 0.5, "comfort": 0.5}
    results = queries.query_20_hotels_by_combined_scores(priorities, city="Tokyo", limit=3)
    for r in results:
        print(f"   - {r['hotel_name']} - Personalized score: {r['personalized_score']:.2f}")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    
    queries.close()


if __name__ == "__main__":
    test_queries()
