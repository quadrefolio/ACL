import re

INTENT_KEYWORDS = {
    "BOOKING_ACTION": [
        r"\bbook(ing)?\b", r"\breserv(e|ation)\b",
        r"\bcancel\b", r"\bmodify\b", r"\bchange\b"
    ],
    "VISA_INFO": [
        r"\bvisa\b", r"\bdo i need\b", r"\brequire(s)? visa\b",
        r"\btravel requirements?\b"
    ],
    "RECOMMEND_HOTEL": [
        r"\bbest\b", r"\btop\b", r"\brecommend(ed)?\b",
        r"\bluxury\b", r"\bhigh(ly)? rated\b"
    ],
    "SEARCH_REVIEW": [
        r"\breview(s)?\b", r"\bfeedback\b", r"\bcomments?\b",
        r"\bwhat do people say\b", r"\bopinions?\b", r"\brating(s)?\b"
    ],
    "HOTEL_SEARCH": [
        r"\bfind\b", r"\bsearch\b", r"\bshow\b", r"\blist\b",
        r"\bhotels?\b", r"\bnear\b", r"\bwhere to stay\b"
    ]
}


def classify_intent(query: str) -> str:
    query = query.lower().strip()

    for intent, patterns in INTENT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, query):
                return intent
    
    return "UNKNOWN"


