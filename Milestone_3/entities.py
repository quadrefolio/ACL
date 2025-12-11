import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Create OpenAI client (official)
client = OpenAI(api_key=API_KEY)

def extract_hotel_entities(query: str) -> dict:
    prompt = f"""
You are an expert travel assistant. Extract hotel-related entities from the USER QUERY.
Return EXACTLY ONE FLAT JSON. NO explanations, no guesses, no assumptions.

REQUIRED JSON KEYS:
"hotels", "cities", "countries", "traveller_type", "demographics"

RULES:

1) "hotels":
    - List only hotel names explicitly mentioned in the query.
    - Do NOT guess the hotel based on description or context.
    - If misspelled but clearly one known hotel matches → correct spelling.
    - If unclear or missing → null.
    - Never invent hotels.

2) "cities" and "countries":
    - List only cities/countries explicitly mentioned in the query.
    - Correct spelling ONLY if the user clearly attempted a real place and there is ONE obvious correction.
    - Do NOT guess cities/countries from context or landmarks.
    - If missing or ambiguous → null.

3) "traveller_type":
    - "family" → mentions children/kids OR 3+ people.
    - "solo" → exactly 1 person.
    - "couple" → 2 adults or romantic context.
    - "business" → work/business context.
    - Otherwise → null.

4) "demographics":
    Must be an object with:
    {{
        "gender": <"male" | "female" | null>,
        "age_group": <"18-24" | "25-34" | "35-44" | "45-54" | "55+" | null>
    }}
    - Gender: detect from words like "he", "she", "father", "mother", "man", "woman". Else null.
    - Age: if exact age given, map to ranges. Else null.

IMPORTANT:
- Do not hallucinate, infer, or guess hotels, cities, or countries from context.
- Correct spelling only if the text clearly matches one real entity.
- Return null if an entity is missing or ambiguous.
- Output only JSON, no extra text, no explanations.

USER QUERY: "{query}"

OUTPUT:
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini", 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        return {"error": "Invalid JSON", "raw": result}
    try:
        return json.loads(match.group(0))
    except:
        return {"error": "Invalid JSON", "raw": result}


# -------------------- TESTS --------------------
if __name__ == "__main__":
    tests = [
        "Show me hotles in Barcelnooa for 2 adlts",
        "Im looking for a buisness-friendly hotel in parees city",
        "Im looking for a buisness-friendly hotel in kwit city",
    ]


    for q in tests:
        print(f"\nQuery: {q}")
        print("Extracted Entities:", extract_hotel_entities(q))
