import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPEN_ROUTER_API")

# Create OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

def extract_hotel_entities(query: str) -> dict:
    prompt = f"""
You EXTRACT hotel-related entities ONLY.
Return EXACTLY ONE FLAT JSON. No explanations.

REQUIRED JSON KEYS:
"hotels", "cities", "countries", "traveller_type", "demographics"

RULES:

1) **hotels**:
    - Must be a LIST of real hotel names from the user text ONLY.
    - If none mentioned → null.

2) **cities & countries**:
    - Extract EXACT mentions. Do NOT guess.
    - If not present → null.

3) **traveller_type**:
    * "family"   → mentions children OR 3+ people.
    * "solo"     → exactly 1 person.
    * "couple"   → 2 adults or romantic context.
    * "business" → business trip context.
    * If unknown → null.

4) **demographics** must be an OBJECT with keys:
    {{
        "gender": <"male" | "female" | null>,
        "age_group": <"18-24" | "25-34" | "35-44" | "45-54" | "55+" | null>
    }}

Gender detection examples:
- "for a man", "guy", "father", "he" → male
- "for a woman", "lady", "mother", "she" → female
If unclear → null.

Age group rules (if mentioned or implied):
- teenager or under 18 → null (ignore, too young)
- If exact age given → convert:
    18–24 → "18-24"
    25–34 → "25-34"
    35–44 → "35-44"
    45–54 → "45-54"
    55+   → "55+"
- If unclear → null.

USER QUERY: "{query}"

OUTPUT ONLY THE JSON OBJECT WITH NO EXTRA TEXT:
"""

    response = client.chat.completions.create(
        model="google/gemma-2-9b-it",
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
        "Show me hotels in Cairo for two adults and one child",
        "Find luxury hotels in Paris for a solo traveler",
        "I need hotels in New York for a business trip",
        "Looking for a romantic getaway in Venice for a couple",
        "Any hotel recommendations?"
    ]

    for q in tests:
        print(f"\nQuery: {q}")
        print("Extracted Entities:", extract_hotel_entities(q))
