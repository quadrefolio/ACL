import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INTENTS = {
    "BOOKING_ACTION": "Booking, reservation, canceling, modifying hotel bookings.",
    "VISA_INFO": "Visa requirements, travel documents.",
    "RECOMMEND_HOTEL": "Recommendations, best or highly rated hotels.",
    "SEARCH_REVIEW": "Reviews, ratings, comments, opinions about hotels.",
    "HOTEL_SEARCH": "Searching or finding hotels, places to stay."
}

def classify_intent(query: str) -> str:
    intents_text = "".join([f"- {k}: {v}\n" for k, v in INTENTS.items()])

    system_prompt = (
        "You are an intent classifier for a travel assistant.\n"
        "Classify the USER MESSAGE into EXACTLY ONE of the following intents:\n\n"
        f"{intents_text}\n"
        "Respond ONLY with the intent key (e.g., BOOKING_ACTION).\n"
        "If it doesn't fit any, return UNKNOWN."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    TEST_QUERIES = [
        "I want to book a room for next weekend.",
        "Do I need a visa to travel to Japan?",
        "Which hotel is the best in Paris?",
        "Show me reviews for Hilton Dubai.",
        "Find me a cheap hotel in Rome.",
        "How's the weather in London?",
        "Cancel my reservation please.",
        "Are there any 5-star hotels near Times Square?"
    ]

    print("=== INTENT CLASSIFICATION TESTS ===\n")

    for query in TEST_QUERIES:
        intent = classify_intent(query)
        print(f"Query: {query}")
        print(f"→ Predicted Intent: {intent}\n")
