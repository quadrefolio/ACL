import os
import openai
from dotenv import load_dotenv

load_dotenv() 

openai.api_key = os.getenv("OPENAI_API_KEY")

INTENTS = {
    "BOOKING_ACTION": "Booking, reservation, canceling, modifying hotel bookings.",
    "VISA_INFO": "Visa requirements, travel documents.",
    "RECOMMEND_HOTEL": "Recommendations, best or highly rated hotels.",
    "SEARCH_REVIEW": "Reviews, ratings, comments, opinions about hotels.",
    "HOTEL_SEARCH": "Searching or finding hotels, places to stay."
}

def classify_intent(query: str) -> str:
    system_prompt = f"""
You are an intent classifier for a travel assistant.
Classify the USER MESSAGE into EXACTLY ONE of the following intents:

{''.join([f"- {k}: {v}\n" for k, v in INTENTS.items()])}

Respond ONLY with the intent key (e.g., BOOKING_ACTION).
If it doesn't fit any, return UNKNOWN.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",      
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0
    )

    intent = response.choices[0].message["content"].strip()
    return intent
