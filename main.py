# =============================
# 🧠 Imports
# =============================
import os
import re
import gc
import json as json_lib
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus
from typing import List

import chromadb
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client

import requests

# =============================
# 🔐 ENV & CLIENTS
# =============================
SUPABASE_URL      = os.environ.get("SUPABASE_URL")
SUPABASE_KEY      = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY")
GOOGLE_CSE_KEY    = os.environ.get("GOOGLE_CSE_KEY")    # Google Custom Search API key
GOOGLE_CSE_ID     = os.environ.get("GOOGLE_CSE_ID")     # Your Programmable Search Engine ID

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise ValueError(
        "Missing required environment variables. "
        "Make sure SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY are set."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# 📚 Trusted Historian Sources
# =============================
TRUSTED_SOURCES = [
    {
        "name": "Ancient Egypt Research Associates (AERA)",
        "url":  "https://aeraweb.org",
        "desc": "Fieldwork and research at Giza by leading Egyptologists",
    },
    {
        "name": "World History Encyclopedia",
        "url":  "https://worldhistory.org",
        "desc": "Peer-reviewed ancient history articles — fully open access",
    },
    {
        "name": "Britannica",
        "url":  "https://britannica.com",
        "desc": "Encyclopedia articles on Ancient Egypt",
    },
    {
        "name": "The British Museum",
        "url":  "https://britishmuseum.org",
        "desc": "One of the world's largest Egyptian artifact collections",
    },
    {
        "name": "The Metropolitan Museum of Art",
        "url":  "https://metmuseum.org",
        "desc": "Extensive Egyptian art and artifact database — fully open",
    },
    {
        "name": "Egyptian Museum Cairo",
        "url":  "https://egyptianmuseum.gov.eg",
        "desc": "Primary Egyptian government museum — direct artifact source",
    },
    {
        "name": "Smithsonian Institution",
        "url":  "https://si.edu",
        "desc": "Authoritative research and collections on ancient civilizations",
    },
    {
        "name": "The Louvre Museum",
        "url":  "https://louvre.fr/en",
        "desc": "Second largest Egyptian collection in the world",
    },
    {
        "name": "UNESCO World Heritage",
        "url":  "https://whc.unesco.org",
        "desc": "Official descriptions of Egyptian heritage sites",
    },
]

TRUSTED_DOMAINS = [s["url"].replace("https://", "") for s in TRUSTED_SOURCES]
# → ["aeraweb.org", "worldhistory.org", "britannica.com", "britishmuseum.org",
#    "metmuseum.org", "egyptianmuseum.gov.eg", "si.edu", "louvre.fr/en", "whc.unesco.org"]

# =============================
# 🔎 Google Custom Search
# =============================
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

def web_search_trusted(query: str, num_results: int = 3) -> list[dict]:
    """
    Search trusted Egyptology sites via Google Custom Search API.
    Returns a list of {title, url, snippet} dicts, or [] on failure.

    Setup (one-time):
      1. Go to https://programmablesearchengine.google.com/
      2. Create a new search engine, add the 5 trusted domains under "Sites to search"
      3. Copy the Search Engine ID → set as GOOGLE_CSE_ID env var
      4. Go to https://console.cloud.google.com/ → APIs → Custom Search JSON API → enable it
      5. Create an API key → set as GOOGLE_CSE_KEY env var
    """
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_ID:
        print("⚠️ GOOGLE_CSE_KEY or GOOGLE_CSE_ID not set — web search fallback disabled")
        return []

    try:
        params = {
            "key": GOOGLE_CSE_KEY,
            "cx":  GOOGLE_CSE_ID,
            "q":   query,
            "num": num_results,
        }
        resp = requests.get(GOOGLE_CSE_ENDPOINT, params=params, timeout=8)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        results = []
        for item in items:
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        print(f"🔍 Web search for '{query}' → {len(results)} results")
        return results
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
        return []


def fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch the text content of a page to give Gemini real grounding context.
    Uses a simple extraction — good enough for article-style pages.
    Falls back to empty string on any error.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PharaohLensBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()

        # Strip HTML tags with a lightweight regex (no BeautifulSoup dependency)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", resp.text, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Trim to max_chars so we don't blow up the Gemini context window
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️ Could not fetch {url}: {e}")
        return ""


def web_search_and_ground(query: str, language: str) -> dict | None:
    """
    Full fallback pipeline:
      1. Search Google CSE for the query across trusted domains
      2. Fetch the top result's page content
      3. Return {url, title, source_name, page_content} for Gemini to use

    Returns None if no useful results found or CSE is not configured.
    """
    results = web_search_trusted(query)
    if not results:
        return None

    # Take the best result (Google CSE already ranks by relevance)
    best = results[0]
    url  = best["url"]

    # Identify which trusted source this belongs to
    source_name = None
    for source in TRUSTED_SOURCES:
        domain = source["url"].replace("https://", "")
        if domain in url:
            source_name = source["name"]
            break
    source_name = source_name or url  # fallback to raw URL if unmatched

    # Fetch the real page content to ground Gemini's answer
    page_content = fetch_page_content(url)
    if not page_content:
        # No content fetched — still return the URL, just without grounding
        page_content = best["snippet"]  # use snippet as minimal context

    return {
        "url":          url,
        "title":        best["title"],
        "source_name":  source_name,
        "page_content": page_content,
    }


# =============================
# 🚦 Usage Limits (non-VIP users)
# =============================
DAILY_MESSAGE_LIMIT = 10
USAGE_TABLE         = "chat_usage"

LIMIT_REACHED_MESSAGE = {
    "en": (
        "📜 You've reached your free daily limit of questions for Horus. "
        "Come back tomorrow for more wisdom from Ancient Egypt — "
        "or upgrade to VIP for unlimited access! 🏺✨"
    ),
    "ar": (
        "📜 لقد وصلت إلى الحد اليومي المجاني من الأسئلة لحورس. "
        "عد غداً لمزيد من حكمة مصر القديمة — "
        "أو قم بالترقية إلى VIP للوصول غير المحدود! 🏺✨"
    ),
}

SERVICE_BUSY_MESSAGE = {
    "en": (
        "📜 Horus is taking a short break right now due to high demand. "
        "Please try again in a few minutes. 🙏"
    ),
    "ar": (
        "📜 حورس يأخذ استراحة قصيرة الآن بسبب الطلب الكبير. "
        "حاول مرة أخرى بعد بضع دقائق. 🙏"
    ),
}

def is_vip_user(user_id: str) -> bool:
    if not user_id:
        return False
    try:
        result = supabase.table("users").select("is_vip").eq("id", user_id).execute()
        if result.data and len(result.data) > 0:
            return bool(result.data[0].get("is_vip", False))
    except Exception as e:
        print(f"⚠️ Could not check VIP status: {e}")
    return False

def get_today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def get_today_usage(user_id: str) -> int:
    try:
        result = (
            supabase.table(USAGE_TABLE)
            .select("message_count")
            .eq("user_id", user_id)
            .eq("usage_date", get_today_utc())
            .execute()
        )
        if result.data and len(result.data) > 0:
            return result.data[0].get("message_count", 0)
    except Exception as e:
        print(f"⚠️ Could not read usage: {e}")
    return 0

def increment_usage(user_id: str):
    try:
        supabase.rpc("increment_chat_usage", {
            "p_user_id": user_id,
            "p_date":    get_today_utc()
        }).execute()
    except Exception as e:
        print(f"⚠️ Could not update usage: {e}")

def check_usage_limit(user_id: str, language: str, is_vip: bool = False) -> str | None:
    """
    Returns a limit-reached message if the user exceeded their daily quota, else None.

    is_vip is passed directly from the client request — the Android app already
    has the correct VIP state from its own auth flow. We do NOT re-query Supabase
    here because:
      1. It adds latency to every single message.
      2. A query failure would silently return False, blocking real VIP users.
      3. The client value is already verified against Supabase at login time.

    Guests (empty user_id) are not tracked server-side.
    VIP users bypass the limit entirely — their usage is also not incremented.
    """
    if not user_id:
        return None   # guest — not tracked
    if is_vip:
        return None   # VIP — no limit, skip everything
    usage = get_today_usage(user_id)
    if usage >= DAILY_MESSAGE_LIMIT:
        return LIMIT_REACHED_MESSAGE.get(language, LIMIT_REACHED_MESSAGE["en"])
    return None


# =============================
# 📝 Request Models
# =============================
class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    language: str = "en"
    user_id: str = ""
    # ✅ FIX: Client sends is_vip directly — avoids a fragile Supabase re-check
    # on every single message. The Android app already knows VIP state reliably.
    # Defaults to False so old app versions keep working without crashing.
    is_vip: bool = False

class QuizRequest(BaseModel):
    chat_history: List[str]
    language: str = "en"
    user_id: str = ""
    is_vip: bool = False  # ✅ same fix for quiz endpoint


# =============================
# 📥 Data Fetching
# =============================
def fetch_data_from_supabase():
    try:
        monuments = supabase.table("monuments").select("*").execute().data
        entities  = supabase.table("entities").select("*").execute().data
        relations = supabase.table("monuments_to_entities").select("*").execute().data
        return monuments, entities, relations
    except Exception as e:
        print(f"❌ Supabase fetch failed: {e}")
        return [], [], []

def build_lookup_maps(entities, relations):
    entity_map = {e["id"]: e for e in entities}
    m_to_e = {}
    for rel in relations:
        m_id, e_id = rel["monument_id"], rel["entity_id"]
        if e_id in entity_map:
            m_to_e.setdefault(m_id, []).append(entity_map[e_id])
    return m_to_e

def build_entity_to_monuments_map(monuments, relations):
    monument_map = {m["id"]: m for m in monuments}
    e_to_m = {}
    for rel in relations:
        m_id, e_id = rel["monument_id"], rel["entity_id"]
        if m_id in monument_map:
            e_to_m.setdefault(e_id, []).append(monument_map[m_id])
    return e_to_m

def parse_json_field(raw):
    try:
        parsed = json_lib.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, dict):
            return "\n".join([f"  - {k}: {v}" for k, v in parsed.items()])
    except:
        pass
    return ""


# =============================
# 📄 Document Builder
# =============================
def build_documents(language: str = "en"):
    monuments, entities, relations = fetch_data_from_supabase()
    m_to_e    = build_lookup_maps(entities, relations)
    e_to_m    = build_entity_to_monuments_map(monuments, relations)
    is_arabic = language == "ar"
    docs = []

    for e in entities:
        entity_id         = e.get("id")
        related_monuments = e_to_m.get(entity_id, [])

        if is_arabic:
            name      = e.get("name_ar")    or e.get("name", "")
            role      = e.get("role_ar")    or e.get("role", "")
            dynasty   = e.get("dynasty_ar") or e.get("dynasty", "")
            json_text = parse_json_field(e.get("json_ar") or e.get("json", "{}"))
            extra     = (e.get("chatbot_extra_ar") or e.get("chatbot_extra") or "").strip()

            monuments_section = ""
            if related_monuments:
                m_lines = []
                for m in related_monuments:
                    m_name    = m.get("class_ar") or m.get("class", "")
                    m_details = m.get("notable_details_ar") or m.get("notable_details", "")
                    m_city    = m.get("city_ar") or m.get("city", "")
                    m_json    = parse_json_field(m.get("json_ar") or m.get("json", "{}"))
                    m_lines.append(f"  • {m_name} — {m_details} (المدينة: {m_city})\n{m_json}")
                monuments_section = (
                    f"\nالمعالم والآثار المرتبطة بـ {name}:\n" + "\n".join(m_lines)
                )

            text = (
                f"شخصية تاريخية: {name}. فرعون: {name}. سيرة ذاتية: {name}.\n"
                f"سجل الشخصية التاريخية — {name}\n\n"
                f"الاسم: {name}\nالدور: {role}\nالأسرة الحاكمة: {dynasty}\n\n"
                f"نبذة تاريخية:\n"
                f"{extra if extra else f'{name} شخصية تاريخية بارزة في الحضارة المصرية القديمة.'}\n\n"
                f"تفاصيل إضافية:\n{json_text}{monuments_section}"
            )
        else:
            name      = e.get("name", "")
            role      = e.get("role", "")
            dynasty   = e.get("dynasty", "")
            json_text = parse_json_field(e.get("json", "{}"))
            extra     = (e.get("chatbot_extra") or "").strip()

            monuments_section = ""
            if related_monuments:
                m_lines = []
                for m in related_monuments:
                    m_name    = m.get("class", "")
                    m_details = m.get("notable_details", "")
                    m_city    = m.get("city", "")
                    m_json    = parse_json_field(m.get("json", "{}"))
                    m_lines.append(f"  • {m_name} — {m_details} (City: {m_city})\n{m_json}")
                monuments_section = (
                    f"\nMONUMENTS & ARTIFACTS ASSOCIATED WITH {name.upper()}:\n"
                    + "\n".join(m_lines)
                )

            text = (
                f"HISTORICAL FIGURE: {name}. PHARAOH: {name}. PERSON: {name}. BIOGRAPHY OF {name}.\n"
                f"HISTORICAL FIGURE RECORD — {name}\n\n"
                f"Name: {name}\nRole: {role}\nDynasty: {dynasty}\n\n"
                f"Historical Summary:\n"
                f"{extra if extra else f'{name} was a prominent figure in Ancient Egyptian history.'}\n\n"
                f"Additional Details:\n{json_text}{monuments_section}"
            )

        docs.append({
            "id":       f"entity_{entity_id}_{language}",
            "text":     text,
            "metadata": {"type": "entity", "name": name, "language": language},
        })

    for row in monuments:
        monument_id      = row.get("id")
        related_entities = m_to_e.get(monument_id, [])

        if is_arabic:
            json_text = parse_json_field(row.get("json_ar") or row.get("json", "{}"))
            m_name    = row.get("class_ar") or row.get("class", "")
            details   = row.get("notable_details_ar") or row.get("notable_details", "")
            city      = row.get("city_ar") or row.get("city", "")
            location  = row.get("current_location_ar") or row.get("currentLocation", "")

            figures_section = ""
            if related_entities:
                fig_names = [(e.get("name_ar") or e.get("name", "")) for e in related_entities]
                figures_section = (
                    f"\nالشخصيات التاريخية المرتبطة بهذا المعلم:\n"
                    + "\n".join(f"  • {n}" for n in fig_names if n)
                )

            text = (
                f"معلم تاريخي: {m_name}. هذه الوثيقة تتحدث عن {m_name}.\n"
                f"سجل المعلم — {m_name}\n\n"
                f"التفاصيل: {details}\nالمدينة: {city}\nالموقع الحالي: {location}\n\n"
                f"وصف تفصيلي:\n{json_text}{figures_section}"
            )
        else:
            json_text = parse_json_field(row.get("json", "{}"))
            m_name    = row.get("class", "")
            details   = row.get("notable_details", "")
            city      = row.get("city", "")
            location  = row.get("currentLocation", "")

            figures_section = ""
            if related_entities:
                fig_names = [e.get("name", "") for e in related_entities]
                figures_section = (
                    f"\nHISTORICAL FIGURES ASSOCIATED WITH THIS MONUMENT:\n"
                    + "\n".join(f"  • {n}" for n in fig_names if n)
                )

            text = (
                f"MONUMENT: {m_name}. This document is about {m_name}.\n"
                f"MONUMENT RECORD — {m_name}\n\n"
                f"Details: {details}\nCity: {city}\nCurrent Location: {location}\n\n"
                f"Detailed Description:\n{json_text}{figures_section}"
            )

        docs.append({
            "id":       f"monument_{monument_id}_{language}",
            "text":     text,
            "metadata": {"type": "monument", "name": m_name, "language": language},
        })

    print(f"✅ Built {len(docs)} documents for language: {language}")
    return docs


# =============================
# 🔢 Gemini Embeddings
# =============================
GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-embedding-001:embedContent"
)

def embed_single(text: str, params: dict, headers: dict) -> List[float]:
    body = {"content": {"parts": [{"text": text}]}, "taskType": "RETRIEVAL_DOCUMENT"}
    for attempt in range(10):
        resp = requests.post(GEMINI_EMBED_URL, params=params, headers=headers, json=body)
        if resp.status_code == 429:
            wait = 12 * (attempt + 1)
            print(f"  ⏳ Rate limit, waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code in (500, 503):
            wait = 5 * (attempt + 1)
            print(f"  ⏳ Gemini server error {resp.status_code}, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
    raise Exception("Embedding failed after 10 retries")

def embed_texts(texts: List[str], batch_size: int = 5) -> List[List[float]]:
    all_embeddings = []
    headers = {"Content-Type": "application/json"}
    params  = {"key": GEMINI_API_KEY}
    total   = len(texts)
    for i, text in enumerate(texts):
        emb = embed_single(text, params, headers)
        all_embeddings.append(emb)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  📦 Embedded {i + 1}/{total}")
        time.sleep(1.5)
    return all_embeddings

def embed_query(text: str) -> List[float]:
    headers = {"Content-Type": "application/json"}
    params  = {"key": GEMINI_API_KEY}
    body    = {
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    resp = requests.post(GEMINI_EMBED_URL, params=params, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


# =============================
# 🔍 Query Expansion
# =============================
QUERY_EXPANSIONS = {
    "ramses":      "Ramesses Ramses Ramosis",
    "ramsis":      "Ramesses Ramses",
    "king tut":    "Tutankhamun Tutankhamen",
    "tutankhamun": "Tutankhamun King Tut boy pharaoh",
    "cleopatra":   "Cleopatra VII queen",
    "nefertiti":   "Nefertiti queen Akhenaten",
    "sphinx":      "Great Sphinx Giza Khafre",
    "pyramids":    "Great Pyramid Giza Khufu Khafre Menkaure",
    "karnak":      "Karnak Temple Luxor Amun",
    "luxor":       "Luxor Temple Karnak Thebes",
    "abu simbel":  "Abu Simbel Ramesses Nubia",
    "valley":      "Valley of the Kings tombs pharaohs Luxor",
    "رمسيس":       "رمسيس الثاني رعمسيس",
    "توت":         "توت عنخ آمون الفرعون الصغير",
    "نفرتيتي":     "نفرتيتي الملكة أخناتون",
    "أبو الهول":   "أبو الهول الجيزة خفرع",
    "الأهرام":     "أهرامات الجيزة خوفو خفرع منقرع",
    "الكرنك":      "معبد الكرنك الأقصر آمون",
}

def expand_query(message: str) -> str:
    query_lower = message.lower()
    extra = []
    for keyword, expansion in QUERY_EXPANSIONS.items():
        if keyword in query_lower:
            extra.append(expansion)
    return f"{message} {' '.join(extra)}" if extra else message


# =============================
# 🧠 Engine Factory
# =============================
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

CACHE_TABLE = "embeddings_cache"

def save_cache_to_supabase(language: str, rows: list):
    try:
        supabase.table(CACHE_TABLE).delete().eq("language", language).execute()
        for i in range(0, len(rows), 50):
            supabase.table(CACHE_TABLE).insert(rows[i:i+50]).execute()
        print(f"💾 Saved {len(rows)} chunks to Supabase cache ({language})")
    except Exception as e:
        print(f"⚠️ Could not save cache: {e}")

def load_cache_from_supabase(language: str):
    try:
        result = supabase.table(CACHE_TABLE).select("*").eq("language", language).execute()
        if result.data and len(result.data) > 0:
            print(f"⚡ Loaded {len(result.data)} cached chunks from Supabase ({language})")
            return result.data
        return None
    except Exception as e:
        print(f"⚠️ Could not load cache: {e}")
        return None

def build_engine(language: str = "en") -> dict:
    print(f"🔄 Building in-memory engine for language: {language}...")
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name=f"pharaoh_{language}")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=f"pharaoh_{language}",
        metadata={"hnsw:space": "cosine"},
    )

    cached = load_cache_from_supabase(language)

    if cached:
        chunked_ids       = [r["chunk_id"]              for r in cached]
        chunked_texts     = [r["text"]                  for r in cached]
        chunked_metadatas = [json_lib.loads(r["metadata"]) for r in cached]
        embeddings        = [json_lib.loads(r["embedding"]) for r in cached]
        collection.add(
            ids=chunked_ids,
            embeddings=embeddings,
            documents=chunked_texts,
            metadatas=chunked_metadatas,
        )
        print(f"✅ {len(cached)} chunks loaded from cache into Chroma ({language})")
    else:
        docs = build_documents(language)
        chunked_texts     = []
        chunked_metadatas = []
        chunked_ids       = []
        for doc in docs:
            chunks = chunk_text(doc["text"])
            for j, chunk in enumerate(chunks):
                chunked_texts.append(chunk)
                chunked_metadatas.append(doc["metadata"])
                chunked_ids.append(f"{doc['id']}_chunk{j}")

        print(f"🔢 Embedding {len(chunked_texts)} chunks via Gemini API...")
        embeddings = embed_texts(chunked_texts)
        collection.add(
            ids=chunked_ids,
            embeddings=embeddings,
            documents=chunked_texts,
            metadatas=chunked_metadatas,
        )
        print(f"✅ {len(chunked_texts)} chunks indexed in Chroma ({language})")
        cache_rows = [
            {
                "chunk_id":  chunked_ids[i],
                "language":  language,
                "text":      chunked_texts[i],
                "metadata":  json_lib.dumps(chunked_metadatas[i]),
                "embedding": json_lib.dumps(embeddings[i]),
            }
            for i in range(len(chunked_ids))
        ]
        save_cache_to_supabase(language, cache_rows)
        del docs, cache_rows
        gc.collect()

    del chunked_texts, chunked_metadatas, chunked_ids, embeddings
    gc.collect()
    return {"collection": collection, "language": language}


# =============================
# 🔍 Retrieval
# =============================
CONFIDENCE_THRESHOLD = 0.45

def retrieve(collection, query: str, top_k: int = 6):
    expanded  = expand_query(query)
    query_emb = embed_query(expanded)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    nodes = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - dist
        if score >= CONFIDENCE_THRESHOLD:
            nodes.append({"text": doc, "metadata": meta, "score": score})
    return nodes


# =============================
# 📝 Prompt Helpers
# =============================
def get_system_prompt(language: str) -> str:
    if language == "ar":
        return (
            "أنت حورس، مرشد متحف الحضارة المصرية القديمة.\n\n"
            "قواعد ثابتة يجب اتباعها دائماً:\n"
            "1. أجب فقط من [سياق المتحف] أدناه إن وُجد وكان كافياً.\n"
            "2. إذا كان السياق غير كافٍ أو فارغاً وتم تزويدك بـ [سياق ويب]:\n"
            "   - استخدم معلومات [سياق ويب] للإجابة.\n"
            "   - لا تذكر اسم الموقع في الإجابة — سيظهر الرابط تلقائياً للمستخدم.\n"
            "3. لا تختلق معلومات أبداً. إذا لم تعرف، قل: 'هذه المعلومة غير متاحة في سجلاتنا.'\n"
            "4. نطاق عملك: مصر القديمة فقط — فراعنة، آلهة، معالم، أسرات، آثار.\n"
            "5. للأسئلة خارج النطاق: قل 'أنا متخصص في مصر القديمة فقط. هل تريد استكشاف فرعون أو معلم أو حضارة؟'\n"
            "6. أولوية اختيار الوثائق:\n"
            "   - سؤال عن شخص/فرعون → وثائق [شخصية تاريخية] أولاً\n"
            "   - سؤال عن مكان/معبد/تمثال → وثائق [معلم أثري] أولاً\n"
            "   - سؤال عن العلاقة بين شخص ومكان → استخدم النوعين\n\n"
            "الأسلوب: دافئ، قصصي، متحمس. اجعل التاريخ حياً. اطرح سؤالاً متابعاً أحياناً.\n"
            "تحدث بالعربية فقط في جميع ردودك."
        )
    return (
        "You are Horus, guide of the Ancient Egyptian Civilization Museum.\n\n"
        "STRICT RULES — follow without exception:\n"
        "1. Answer ONLY from the [Museum Context] section when it contains relevant information.\n"
        "2. If context is empty or insufficient but [Web Context] is provided:\n"
        "   - Use the [Web Context] to answer the question accurately.\n"
        "   - Do NOT mention the source site name in your answer — the link will be shown automatically.\n"
        "3. Never fabricate. If genuinely unknown, say: 'This detail isn't in our records.'\n"
        "4. Scope: Ancient Egypt only — pharaohs, gods, monuments, dynasties, artifacts.\n"
        "5. Off-topic questions: say 'I can only guide you through Ancient Egypt! "
        "What would you like to explore — a pharaoh, a monument, or a dynasty?'\n"
        "6. Document priority:\n"
        "   - Person/pharaoh question → prioritize [HISTORICAL FIGURE] documents first\n"
        "   - Place/temple/statue question → prioritize [MONUMENT] documents first\n"
        "   - Relationship between person and place → use both\n\n"
        "Style: Warm, storytelling, enthusiastic. Make history feel alive.\n"
        "Occasionally ask a follow-up like 'Would you like to know more about this?'\n"
        "Respond in English only."
    )

def build_context_block(nodes: list, language: str) -> str:
    if not nodes:
        return ""
    chunks = []
    for i, node in enumerate(nodes, 1):
        doc_type = node["metadata"].get("type", "document")
        doc_name = node["metadata"].get("name", "")
        score    = round(node["score"], 3)
        if language == "ar":
            type_label = "شخصية تاريخية" if doc_type == "entity" else "معلم أثري"
        else:
            type_label = "HISTORICAL FIGURE" if doc_type == "entity" else "MONUMENT"
        label = f"[{i}] {type_label}: {doc_name} (score: {score})"
        chunks.append(f"{label}\n{node['text'].strip()}")

    sep = "─" * 40
    if language == "ar":
        header = "[سياق المتحف — مصدرك الحصري للمعلومات]"
        footer = "استخدم هذا السياق فقط للإجابة على السؤال التالي."
    else:
        header = "[Museum Context — your ONLY allowed information source]"
        footer = "Use ONLY the above context to answer the question that follows."

    return (
        f"{header}\n{sep}\n\n"
        + f"\n\n{sep}\n\n".join(chunks)
        + f"\n\n{sep}\n{footer}"
    )

def build_web_context_block(page_content: str, language: str) -> str:
    """Build a context block from fetched web page content."""
    sep = "─" * 40
    if language == "ar":
        header = "[سياق ويب — معلومات من مصدر موثوق]"
        footer = "استخدم هذا السياق للإجابة على السؤال التالي."
    else:
        header = "[Web Context — information from a trusted historical source]"
        footer = "Use the above web context to answer the question that follows."

    return f"{header}\n{sep}\n\n{page_content.strip()}\n\n{sep}\n{footer}"


# =============================
# 🤖 Gemini LLM via REST
# =============================
GEMINI_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

def gemini_generate(prompt: str, system: str = "", history: list = None) -> str:
    headers  = {"Content-Type": "application/json"}
    params   = {"key": GEMINI_API_KEY}
    contents = []

    last_role = None
    for turn in (history or []):
        role = turn.get("role", "user")
        msg  = turn.get("content", "").strip()
        if not msg:
            continue
        if role == last_role:
            continue
        contents.append({"role": role, "parts": [{"text": msg}]})
        last_role = role

    if contents and contents[-1]["role"] == "user":
        contents[-1]["parts"][0]["text"] += "\n" + prompt
    else:
        contents.append({"role": "user", "parts": [{"text": prompt}]})

    body = {
        "contents": contents,
        "generationConfig": {"temperature": 0.1},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    resp = requests.post(GEMINI_CHAT_URL, params=params, headers=headers, json=body)
    if not resp.ok:
        print(f"❌ Gemini error {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# =============================
# 💬 Chat Logic
# =============================
def run_chat(engine: dict, message: str, history: List[dict]) -> str:
    collection = engine["collection"]
    language   = engine["language"]

    # ── Step 1: Try RAG first ─────────────────────────────────────────────────
    nodes         = retrieve(collection, message)
    context_block = build_context_block(nodes, language)

    system_prompt  = get_system_prompt(language)
    recent_history = history[-12:] if len(history) > 12 else history
    gemini_history = []
    for turn in recent_history:
        r   = turn.get("role", "user")
        msg = turn.get("content", "")
        if r in ("user", "assistant") and msg:
            gemini_history.append({
                "role":    "user" if r == "user" else "model",
                "content": msg,
            })

    # ── Step 2: RAG has enough context → answer directly ─────────────────────
    if context_block:
        augmented = f"{context_block}\n\nQuestion: {message}"
        answer    = gemini_generate(
            prompt=augmented,
            system=system_prompt,
            history=gemini_history,
        )
        return answer

    # ── Step 3: RAG found nothing → use real web search as fallback ───────────
    print(f"📭 RAG miss for: '{message}' — triggering web search fallback")

    # Build a clean English search query even for Arabic questions
    # so that search results are in English (better coverage on these sites)
    search_query = f"Ancient Egypt {message}"

    web_result = web_search_and_ground(search_query, language)

    if web_result:
        # Ground Gemini on the real fetched page content
        web_context = build_web_context_block(web_result["page_content"], language)
        augmented   = f"{web_context}\n\nQuestion: {message}"

        answer = gemini_generate(
            prompt=augmented,
            system=system_prompt,
            history=gemini_history,
        )

        # Append the real source link — this is the exact article page, not a search page
        url         = web_result["url"]
        source_name = web_result["source_name"]
        title       = web_result["title"]

        if language == "ar":
            answer += f"\n\n📖 المصدر: [{title}]({url})"
        else:
            answer += f"\n\n📖 Source: [{title}]({url})"

    else:
        # Web search unavailable or returned nothing — answer from Gemini training only
        # with a clear disclaimer (no fake links)
        print("⚠️ Web search returned no results — answering from Gemini training data only")
        augmented = message
        answer    = gemini_generate(
            prompt=augmented,
            system=system_prompt,
            history=gemini_history,
        )
        if language == "ar":
            answer += "\n\n📜 *هذه المعلومة مستندة إلى المعرفة العامة ولم يُتحقق منها من مصدر موثوق.*"
        else:
            answer += "\n\n📜 *This answer is based on general knowledge and has not been verified from a trusted source.*"

    return answer


# =============================
# 🚀 FastAPI App
# =============================
app      = FastAPI()
_engines: dict = {}
_engine_lock   = threading.Lock()

def get_engine(language: str) -> dict:
    if language in _engines:
        return _engines[language]
    with _engine_lock:
        if language not in _engines:
            _engines[language] = build_engine(language)
            print(f"✅ Engine ready: {language}")
    return _engines[language]

def build_engines_sequentially():
    time.sleep(3)
    print("🚀 Background: building EN engine...")
    get_engine("en")
    gc.collect()
    print("✅ EN ready. Waiting before AR...")
    time.sleep(10)
    print("🚀 Background: building AR engine...")
    get_engine("ar")
    print("✅ Both engines ready.")

threading.Thread(target=build_engines_sequentially, daemon=True).start()


# =============================
# 🌐 Endpoints
# =============================
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status":      "ok",
        "en_engine":   "ready" if "en" in _engines else "building...",
        "ar_engine":   "ready" if "ar" in _engines else "building...",
        "web_search":  "enabled" if (GOOGLE_CSE_KEY and GOOGLE_CSE_ID) else "disabled (set GOOGLE_CSE_KEY + GOOGLE_CSE_ID)",
    }

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "ok", "message": "PharaohLens API Active 👑"}

@app.post("/chat")
def chat(request: ChatRequest):
    # ✅ Pass is_vip from the request — no Supabase re-fetch needed
    limit_message = check_usage_limit(request.user_id, request.language, request.is_vip)
    if limit_message:
        return {"answer": limit_message}

    try:
        engine = get_engine(request.language)
        answer = run_chat(engine, request.message, request.history)

        # ✅ Only increment usage for non-VIP logged-in users
        if request.user_id and not request.is_vip:
            increment_usage(request.user_id)

        return {"answer": answer}

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        print(f"❌ Gemini HTTP error: {status} — {e}")
        return {"answer": SERVICE_BUSY_MESSAGE.get(request.language, SERVICE_BUSY_MESSAGE["en"])}

    except Exception as e:
        print(f"❌ Chat error: {e}")
        return {"answer": SERVICE_BUSY_MESSAGE.get(request.language, SERVICE_BUSY_MESSAGE["en"])}

@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
    # ✅ Pass is_vip from the request — same fix as /chat
    limit_message = check_usage_limit(request.user_id, request.language, request.is_vip)
    if limit_message:
        return {"quiz": [{
            "question": limit_message,
            "options": ["OK", "OK", "OK", "OK"],
            "correct_answer": "OK",
        }]}

    try:
        history_text = (
            "\n".join(request.chat_history)
            if request.chat_history
            else "General Ancient Egyptian history, pharaohs, and monuments"
        )

        lang_instruction = (
            "CRITICAL: Generate ALL questions, options, and answers in Arabic only."
            if request.language == "ar"
            else "Generate all questions and answers in English."
        )

        quiz_prompt = f"""Based on this conversation about Ancient Egypt:
{history_text}

Generate 5 multiple choice questions about the topics discussed.
{lang_instruction}

Rules:
- Return ONLY a raw JSON array. No markdown, no extra text, no backticks.
- Each object must have exactly: "question", "options" (list of exactly 4 strings), "correct_answer".

Example format:
[
  {{
    "question": "Who built the Great Pyramid?",
    "options": ["Khufu", "Khafre", "Menkaure", "Ramesses II"],
    "correct_answer": "Khufu"
  }}
]"""

        raw  = gemini_generate(quiz_prompt).strip().replace("```json", "").replace("```", "").strip()
        quiz = json_lib.loads(raw)

        # ✅ Only increment for non-VIP logged-in users
        if request.user_id and not request.is_vip:
            increment_usage(request.user_id)

        return {"quiz": quiz}

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        print(f"❌ Gemini HTTP error (quiz): {status} — {e}")
        return {"quiz": [{
            "question": SERVICE_BUSY_MESSAGE.get(request.language, SERVICE_BUSY_MESSAGE["en"]),
            "options": ["OK", "OK", "OK", "OK"],
            "correct_answer": "OK",
        }]}

    except Exception as e:
        print(f"❌ Quiz error: {e}")
        return {"quiz": [{
            "question": SERVICE_BUSY_MESSAGE.get(request.language, SERVICE_BUSY_MESSAGE["en"]),
            "options": ["OK", "OK", "OK", "OK"],
            "correct_answer": "OK",
        }]}

@app.api_route("/rebuild-index", methods=["GET", "POST"])
def rebuild_index(background_tasks: BackgroundTasks):
    global _engines
    if getattr(app.state, "rebuilding", False):
        return {"error": "Rebuild already in progress."}

    app.state.rebuilding = True
    _engines = {}
    gc.collect()

    def do_rebuild():
        try:
            try:
                supabase.table(CACHE_TABLE).delete().neq("chunk_id", "").execute()
                print("🗑️ Supabase embedding cache cleared")
            except Exception as e:
                print(f"⚠️ Could not clear cache: {e}")
            build_engines_sequentially()
        finally:
            app.state.rebuilding = False

    background_tasks.add_task(do_rebuild)
    return {"message": "Rebuild started. Cache cleared, re-embedding from scratch.", "status": "rebuilding"}

@app.api_route("/rebuild-status", methods=["GET"])
def rebuild_status():
    return {
        "is_rebuilding":  getattr(app.state, "rebuilding", False),
        "en_engine":      "ready" if "en" in _engines else "building...",
        "ar_engine":      "ready" if "ar" in _engines else "building...",
        "engines_loaded": list(_engines.keys()),
    }

@app.api_route("/debug-retrieve", methods=["GET"])
def debug_retrieve(q: str = "Ramses II", lang: str = "en"):
    try:
        engine  = get_engine(lang)
        nodes   = retrieve(engine["collection"], q)
        results = [
            {
                "rank":    i + 1,
                "score":   node["score"],
                "type":    node["metadata"].get("type"),
                "name":    node["metadata"].get("name"),
                "preview": node["text"][:300],
            }
            for i, node in enumerate(nodes)
        ]
        return {"query": q, "language": lang, "results": results}
    except Exception as e:
        return {"error": str(e)}

@app.api_route("/debug-entity", methods=["GET"])
def debug_entity(name: str = "Ramesses II"):
    try:
        result = supabase.table("entities").select("*").ilike("name", f"%{name}%").execute()
        if not result.data:
            return {"error": "No entity found"}
        entity = result.data[0]
        return {
            "all_keys":         list(entity.keys()),
            "name":             entity.get("name"),
            "name_ar":          entity.get("name_ar"),
            "chatbot_extra":    entity.get("chatbot_extra"),
            "chatbot_extra_ar": entity.get("chatbot_extra_ar"),
        }
    except Exception as e:
        return {"error": str(e)}

# ── New: debug web search so you can test it without going through chat ───────
@app.api_route("/debug-websearch", methods=["GET"])
def debug_websearch(q: str = "Imhotep architect", lang: str = "en"):
    """
    Test the web search fallback independently.
    Call: GET /debug-websearch?q=Imhotep+architect&lang=en
    """
    result = web_search_and_ground(f"Ancient Egypt {q}", lang)
    if not result:
        return {"error": "No results found or web search not configured"}
    return {
        "url":             result["url"],
        "title":           result["title"],
        "source_name":     result["source_name"],
        "page_content":    result["page_content"][:500] + "...",
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
