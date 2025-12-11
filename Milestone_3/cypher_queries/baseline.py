# --- TASK 1: BOOKING & VISA ASSISTANT (The "Facts") ---
SEARCH_QUERIES = {
    # 1. Basic Availability: Find hotels in a specific city
    "FIND_BY_CITY": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City) 
        WHERE toLower(c.name) CONTAINS toLower($city) 
        RETURN h.name as Hotel, h.star_rating as Stars, h.average_reviews_score as Rating
        LIMIT 5
    """,
    
    # 2. Quality Filter: Strict star rating search
    "FIND_BY_RATING": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE h.star_rating >= $rating AND toLower(c.name) CONTAINS toLower($city)
        RETURN h.name as Hotel, h.star_rating as Stars
        ORDER BY h.star_rating DESC
    """,
    
    # 3. Visa Check: Specific rule between two countries
    "CHECK_VISA_REQUIREMENT": """
        MATCH (from:Country {name: $from_country}), (to:Country {name: $to_country})
        OPTIONAL MATCH (from)-[r:NEEDS_VISA]->(to) 
        RETURN to.name as Destination, 
               CASE WHEN r IS NULL THEN 'Visa-Free' ELSE r.visa_type END as Visa_Type
    """,
    
    # 4. "Can I go?": Find hotels in a country ONLY if No Visa is required
    "FIND_VISA_FREE_DESTINATIONS": """
        MATCH (me:Country {name: $from_country})
        MATCH (dest:Country)
        WHERE NOT (me)-[:NEEDS_VISA]->(dest)
        MATCH (dest)<-[:LOCATED_IN]-(c:City)<-[:LOCATED_IN]-(h:Hotel)
        RETURN h.name as Hotel, c.name as City, dest.name as Country
        LIMIT 5
    """,
    
    # 5. Facility Search: For booking specific needs
    "FIND_HIGH_FACILITIES": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE h.facilities_base >= 9.0 AND toLower(c.name) = toLower($city)
        RETURN h.name as Hotel, h.facilities_base as Facility_Score
    """,
    
    # 6. Cleanliness Priority: Critical for booking decisions
    "FIND_HIGH_CLEANLINESS": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE h.cleanliness_base >= 9.0 
        RETURN h.name as Hotel, c.name as City, h.cleanliness_base as Score
        ORDER BY Score DESC LIMIT 5
    """,
    
    # 7. Value for Money: Budget-conscious booking
    "FIND_BEST_VALUE": """
        MATCH (h:Hotel)
        WHERE h.star_rating >= 4
        RETURN h.name as Hotel, h.average_reviews_score as Value_Score
        ORDER BY Value_Score DESC LIMIT 5
    """,
    
    # 8. Visa Type Info: Get details on the specific visa needed
    "GET_VISA_DETAILS": """
        MATCH (from:Country {name: $from_country})-[r:NEEDS_VISA]->(to:Country)
        RETURN to.name as Destination, r.visa_type as Visa_Type
        LIMIT 5
    """,
    
    # 9. Geography Search: Find all hotels in a specific Country  useless
    "FIND_HOTELS_IN_COUNTRY": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(cnt:Country) 
        WHERE toLower(cnt.name) = toLower($country) 
        RETURN h.name as Hotel, c.name as City
        LIMIT 5
    """,
    
    # 10. Specific Hotel Lookup: For confirming a booking details
    "GET_HOTEL_DETAILS": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE toLower(h.name) CONTAINS toLower($hotel_name) 
        RETURN h.name as hotel_name, h.star_rating, c.name as city, h.cleanliness_base, h.embedding_minilm IS NOT NULL as has_vector
    """,

    # 11. Review Search (New for SEARCH_REVIEW intent)
    "GET_HOTEL_REVIEWS": """
        MATCH (r:Review)-[:REVIEWED]->(h:Hotel)
        WHERE toLower(h.name) CONTAINS toLower($hotel_name)
        RETURN r.text as Review, r.score_overall as Score
        ORDER BY r.score_overall DESC LIMIT 5
    """
}

# --- TASK 2: HOTEL RECOMMENDER SYSTEM (The "Suggestions") ---
RECOMMENDATION_QUERIES = {
    # 1. Collaborative Filtering: "People who stayed here also stayed..."
    "REC_SIMILAR_USERS": """
        MATCH (t:Traveller)-[:STAYED_AT]->(h:Hotel {name: $hotel_name})
        MATCH (t)-[:STAYED_AT]->(rec:Hotel)
        WHERE rec <> h
        RETURN rec.name as Recommendation, count(t) as Shared_Visitors
        ORDER BY Shared_Visitors DESC LIMIT 5
    """,
    
    # 2. Demographic: "Best hotels for Families/Couples"
    "REC_BY_TRAVELLER_TYPE": """
        MATCH (t:Traveller {type: $type})-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
        RETURN h.name as Hotel, avg(r.score_overall) as Avg_Rating_By_Group
        ORDER BY Avg_Rating_By_Group DESC LIMIT 5
    """,
    
    # 3. Solo Female Safety: High location scores from Solo Female travellers
    "REC_SAFE_SOLO_FEMALE": """
        MATCH (t:Traveller {gender: 'Female', type: 'Solo'})-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
        RETURN h.name as Hotel, avg(r.score_location) as Safety_Location_Score
        ORDER BY Safety_Location_Score DESC LIMIT 5
    """,
    
    # 4. Hidden Gems: Low Stars (<=3) but High Reviews (>=9.0)
    "REC_HIDDEN_GEMS": """
        MATCH (h:Hotel) 
        WHERE h.star_rating <= 3 AND h.average_reviews_score >= 9.0 
        RETURN h.name as Hidden_Gem, h.star_rating, h.average_reviews_score
    """,
    
    # 5. Better Alternatives: Same city, better rating
    "REC_UPGRADE_STAY": """
        MATCH (h:Hotel {name: $current_hotel})-[:LOCATED_IN]->(c:City)<-[:LOCATED_IN]-(upgrade:Hotel)
        WHERE upgrade.star_rating > h.star_rating
        RETURN upgrade.name as Better_Option, upgrade.star_rating as Stars
    """,
    
    # 6. National Favorites: Where do people from my country go?
    "REC_BY_ORIGIN_COUNTRY": """
        MATCH (t:Traveller)-[:FROM_COUNTRY]->(c:Country {name: $my_country})
        MATCH (t)-[:STAYED_AT]->(h:Hotel)
        RETURN h.name as Hotel, count(t) as Visits_From_Compatriots
        ORDER BY Visits_From_Compatriots DESC LIMIT 5
    """,
    
    # 7. Consistent Quality: "Exceeds Expectations" (Base < Review Score)
    "REC_EXCEEDS_EXPECTATIONS": """
        MATCH (h:Hotel)
        WHERE h.average_reviews_score > (h.star_rating * 2) 
        RETURN h.name as Hotel, h.star_rating, h.average_reviews_score
        ORDER BY h.average_reviews_score DESC LIMIT 5
    """,
    
    # 8. Comfort Seekers: Sort by comfort base score
    "REC_HIGH_COMFORT": """
        MATCH (h:Hotel) 
        WHERE h.comfort_base > 9.0 
        RETURN h.name as Hotel, h.comfort_base
        ORDER BY h.comfort_base DESC LIMIT 5
    """,
    
    # 9. Popular City Discovery: Most visited cities in the graph
    "REC_POPULAR_CITIES": """
        MATCH (t:Traveller)-[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(c:City)
        RETURN c.name as City, count(t) as Visitor_Count
        ORDER BY Visitor_Count DESC LIMIT 3
    """,
    
    # 10. The "Perfect" Stay: High scores across ALL categories
    "REC_PERFECT_STAY": """
        MATCH (h:Hotel)
        WHERE h.cleanliness_base > 9.0 AND h.comfort_base > 9.0 AND h.facilities_base > 9.0
        RETURN h.name as Perfect_Hotel
        LIMIT 5
    """
}

def resolve_cypher_query(intent, entities):
    """
    Logic Router: Maps your specific Intent + Entities to the best Cypher Template.
    Returns: (query_string, params_dict)
    """
    params = {}
    
    # 1. Unpack Entity Lists (Taking the first match as primary)
    hotel = entities.get("hotels")[0] if entities.get("hotels") else None
    city = entities.get("cities")[0] if entities.get("cities") else None
    country = entities.get("countries")[0] if entities.get("countries") else None
    
    # Travellers & Demographics
    t_type = entities.get("traveller_type")
    demos = entities.get("demographics", {})
    gender = demos.get("gender") if demos else None
    
    # --- ROUTING LOGIC ---

    # A. HOTEL SEARCH
    if intent == "HOTEL_SEARCH":
        if city:
            params = {"city": city}
            return SEARCH_QUERIES["FIND_BY_CITY"], params
        elif country:
            params = {"country": country}
            return SEARCH_QUERIES["FIND_HOTELS_IN_COUNTRY"], params
        elif hotel:
            params = {"hotel_name": hotel}
            return SEARCH_QUERIES["GET_HOTEL_DETAILS"], params
        else:
            # Fallback if no location specified
            return SEARCH_QUERIES["FIND_HIGH_CLEANLINESS"], {}

    # B. VISA INFO
    elif intent == "VISA_INFO":
        countries = entities.get("countries", [])
        if len(countries) >= 2:
            # Assume first is From, second is To
            params = {"from_country": countries[0], "to_country": countries[1]}
            return SEARCH_QUERIES["CHECK_VISA_REQUIREMENT"], params
        elif len(countries) == 1:
            # Assume user asks "Visa for Japan" (from a default origin)
            params = {"from_country": "United States", "to_country": countries[0]} 
            return SEARCH_QUERIES["GET_VISA_DETAILS"], params
        else:
            # General visa-free query
            params = {"from_country": "United States"} # Defaulting
            return SEARCH_QUERIES["FIND_VISA_FREE_DESTINATIONS"], params

    # C. RECOMMENDATIONS
    elif intent == "RECOMMEND_HOTEL":
        # Specific Demographic Rule
        if gender == "female" and t_type == "solo":
            return RECOMMENDATION_QUERIES["REC_SAFE_SOLO_FEMALE"], {}
        
        elif t_type:
            params = {"type": t_type.capitalize()} # Capitalize to match CSV (e.g. 'Family')
            return RECOMMENDATION_QUERIES["REC_BY_TRAVELLER_TYPE"], params
            
        elif hotel:
            params = {"hotel_name": hotel}
            return RECOMMENDATION_QUERIES["REC_SIMILAR_USERS"], params
            
        elif country:
            params = {"my_country": country}
            return RECOMMENDATION_QUERIES["REC_BY_ORIGIN_COUNTRY"], params
            
        else:
            return RECOMMENDATION_QUERIES["REC_POPULAR_CITIES"], {}

    # D. REVIEWS
    elif intent == "SEARCH_REVIEW":
        if hotel:
            params = {"hotel_name": hotel}
            return SEARCH_QUERIES["GET_HOTEL_REVIEWS"], params
        else:
            return SEARCH_QUERIES["FIND_HIGH_CLEANLINESS"], {} # Fallback

    # E. BOOKING
    elif intent == "BOOKING_ACTION":
        if hotel:
            params = {"hotel_name": hotel}
            return SEARCH_QUERIES["GET_HOTEL_DETAILS"], params
        else:
            # If they want to book but didn't say where, maybe find cities?
            return SEARCH_QUERIES["FIND_HIGH_FACILITIES"], {"city": city or "London"}

    return None, {}