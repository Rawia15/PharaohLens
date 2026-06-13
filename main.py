# =============================
# 🧠 Imports
# =============================
import os
import gc
import json as json_lib
import threading
import time
from typing import List

import chromadb
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client

import requests

# =============================
# 🔐 ENV & CLIENTS
# =============================
SUPABASE_URL  = os.environ.get("SUPABASE_URL")
SUPABASE_KEY  = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise ValueError(
        "Missing required environment variables. "
        "Make sure SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY are set."
    )

# Gemini via REST API — no SDK needed
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
        "desc": "Peer-reviewed ancient history articles",
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
        "desc": "Extensive Egyptian art and artifact database",
    },
]

SOURCES_BY_KEYWORD = {
    "aera":                              TRUSTED_SOURCES[0],
    "aeraweb":                           TRUSTED_SOURCES[0],
    "ancient egypt research associates": TRUSTED_SOURCES[0],
    "world history":                     TRUSTED_SOURCES[1],
    "worldhistory":                      TRUSTED_SOURCES[1],
    "britannica":                        TRUSTED_SOURCES[2],
    "british museum":                    TRUSTED_SOURCES[3],
    "britishmuseum":                     TRUSTED_SOURCES[3],
    "metropolitan":                      TRUSTED_SOURCES[4],
    "metmuseum":                         TRUSTED_SOURCES[4],
}

def detect_cited_source(answer: str) -> dict | None:
    """Find which trusted source Horus mentioned in his answer."""
    answer_lower = answer.lower()
    for keyword, source in SOURCES_BY_KEYWORD.items():
        if keyword in answer_lower:
            return source
    return None

# =============================
# 📝 Request Models
# =============================
class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    language: str = "en"

class QuizRequest(BaseModel):
    chat_history: List[str]
    language: str = "en"

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
    """Returns list of (text, metadata) tuples — no heavy objects."""
    monuments, entities, relations = fetch_data_from_supabase()
    m_to_e    = build_lookup_maps(entities, relations)
    e_to_m    = build_entity_to_monuments_map(monuments, relations)
    is_arabic = language == "ar"
    docs = []  # list of {"text": str, "metadata": dict, "id": str}

    # ── Entity documents ─────────────────────────────────────────────────────
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

    # ── Monument documents ────────────────────────────────────────────────────
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
GEMINI_BATCH_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-embedding-001:batchEmbedContents"
)

def embed_texts(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """Embed texts using Gemini REST API in batches."""
    all_embeddings = []
    headers = {"Content-Type": "application/json"}
    params  = {"key": GEMINI_API_KEY}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        body  = {
            "requests": [
                {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": t}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                }
                for t in batch
            ]
        }
        resp = requests.post(GEMINI_BATCH_EMBED_URL, params=params, headers=headers, json=body)
        resp.raise_for_status()
        for emb in resp.json()["embeddings"]:
            all_embeddings.append(emb["values"])
        print(f"  📦 Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        time.sleep(0.5)  # stay within free-tier rate limits
    return all_embeddings

def embed_query(text: str) -> List[float]:
    """Embed a single query using Gemini REST API."""
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
    # English aliases / common misspellings
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
    # Arabic aliases
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
# 🧠 Engine Factory (in-memory Chroma)
# =============================
def build_engine(language: str = "en") -> dict:
    print(f"🔄 Building in-memory engine for language: {language}...")

    # In-memory Chroma client — no disk, no persistence issues on Railway
    chroma_client = chromadb.Client()
    collection    = chroma_client.create_collection(
        name=f"pharaoh_{language}",
        metadata={"hnsw:space": "cosine"},
    )

    docs = build_documents(language)

    # Embed in batches and insert into Chroma
    texts     = [d["text"]     for d in docs]
    metadatas = [d["metadata"] for d in docs]
    ids       = [d["id"]       for d in docs]

    print(f"🔢 Embedding {len(texts)} documents via Gemini API...")
    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"✅ {len(docs)} documents indexed in Chroma ({language})")

    # Free memory before building next language
    del docs, texts, metadatas, ids, embeddings
    gc.collect()

    # LLM called via gemini_generate() REST helper

    return {
        "collection": collection,
        "llm":        llm,
        "language":   language,
    }

# =============================
# 🔍 Retrieval
# =============================
CONFIDENCE_THRESHOLD = 0.45

def retrieve(collection, query: str, top_k: int = 6):
    """Embed query and search Chroma. Returns list of result dicts."""
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
        # Chroma cosine distance → similarity score (1 - distance)
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
            "2. إذا كان السياق غير كافٍ أو فارغاً:\n"
            "   - ابدأ ردك بـ '📜 استناداً إلى المصادر التاريخية،'\n"
            "   - أجب من معرفتك العامة بمصر القديمة.\n"
            "   - اذكر اسم المصدر الأنسب داخل الإجابة بشكل طبيعي، مثل: "
            "'وفقاً لدائرة المعارف بريتانيكا...' أو 'كما يوثق المتحف البريطاني...' "
            "أو 'تشير أبحاث جمعية أبحاث مصر القديمة (AERA) إلى...'.\n"
            "   - المصادر التي يمكنك الاستناد إليها: "
            "جمعية أبحاث مصر القديمة AERA (aeraweb.org)، "
            "موسوعة التاريخ العالمي (worldhistory.org)، "
            "بريتانيكا (britannica.com)، "
            "المتحف البريطاني (britishmuseum.org)، "
            "متحف متروبوليتان للفنون (metmuseum.org).\n"
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
        "2. If context is empty or insufficient:\n"
        "   - Begin your reply with the EXACT marker '📜 Based on historical records,'\n"
        "   - Answer from your general knowledge about Ancient Egypt only.\n"
        "   - Naturally cite the most relevant source by name inside your answer, e.g.: "
        "'According to Britannica...', 'As documented by the British Museum...', "
        "or 'AERA fieldwork at Giza suggests...'.\n"
        "   - Trusted sources you may cite: "
        "Ancient Egypt Research Associates AERA (aeraweb.org), "
        "World History Encyclopedia (worldhistory.org), "
        "Britannica (britannica.com), "
        "The British Museum (britishmuseum.org), "
        "The Metropolitan Museum of Art (metmuseum.org).\n"
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

# =============================
# 🤖 Gemini LLM via REST
# =============================
GEMINI_CHAT_URL = (
    "https://generativelanguage.googleapis.com/v1/models/"
    "gemini-1.5-flash:generateContent"
)

def gemini_generate(prompt: str, system: str = "", history: list = None) -> str:
    """Call Gemini 1.5 Flash via REST API."""
    headers = {"Content-Type": "application/json"}
    params  = {"key": GEMINI_API_KEY}

    contents = []

    # Add history turns
    for turn in (history or []):
        contents.append({
            "role": turn["role"],
            "parts": [{"text": turn["content"]}]
        })

    # Add current user message
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    body = {
        "contents": contents,
        "generationConfig": {"temperature": 0.1},
    }

    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    resp = requests.post(GEMINI_CHAT_URL, params=params, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

# =============================
# 💬 Chat Logic
# =============================
def run_chat(engine: dict, message: str, history: List[dict]) -> str:
    collection = engine["collection"]
    llm        = engine["llm"]
    language   = engine["language"]

    # 1. Retrieve relevant nodes
    nodes = retrieve(collection, message)

    # 2. Build context block
    context_block = build_context_block(nodes, language)

    # 3. Augment user message
    augmented = (
        f"{context_block}\n\nQuestion: {message}"
        if context_block
        else message
    )

    # 4. Call Gemini via REST
    system_prompt  = get_system_prompt(language)
    recent_history = history[-12:] if len(history) > 12 else history

    gemini_history = []
    for turn in recent_history:
        r    = turn.get("role", "user")
        msg  = turn.get("content", "")
        if r in ("user", "assistant") and msg:
            gemini_history.append({
                "role":    "user" if r == "user" else "model",
                "content": msg,
            })

    answer = gemini_generate(
        prompt=augmented,
        system=system_prompt,
        history=gemini_history,
    )

    # If Horus fell back to historian knowledge, attach the source he cited
    fallback_markers = ["📜 Based on historical records", "📜 استناداً إلى المصادر التاريخية"]
    is_fallback = any(marker in answer for marker in fallback_markers)

    if is_fallback:
        source = detect_cited_source(answer)
        if source:
            if language == "ar":
                answer += f"\n\n📌 المصدر: [{source['name']}]({source['url']})"
            else:
                answer += f"\n\n📌 Source: [{source['name']}]({source['url']})"

    return answer

# =============================
# 🚀 FastAPI App
# =============================
app      = FastAPI()
_engines: dict        = {}
_engine_lock          = threading.Lock()

def get_engine(language: str) -> dict:
    if language in _engines:
        return _engines[language]
    with _engine_lock:
        if language not in _engines:
            _engines[language] = build_engine(language)
            print(f"✅ Engine ready: {language}")
    return _engines[language]

def build_engines_sequentially():
    """Build EN first, GC, then AR — keeps peak memory flat."""
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
        "status":    "ok",
        "en_engine": "ready" if "en" in _engines else "building...",
        "ar_engine": "ready" if "ar" in _engines else "building...",
    }

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return {"status": "ok", "message": "PharaohLens API Active 👑"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        engine = get_engine(request.language)
        answer = run_chat(engine, request.message, request.history)
        return {"answer": answer}
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return {"error": str(e)}

@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest):
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

        raw = gemini_generate(quiz_prompt).strip().replace("```json", "").replace("```", "").strip()
        quiz     = json_lib.loads(raw)
        return {"quiz": quiz}

    except Exception as e:
        return {"error": f"Quiz Error: {str(e)}"}

@app.api_route("/rebuild-index", methods=["GET", "POST"])
def rebuild_index(background_tasks: BackgroundTasks):
    """Wipe engines and rebuild from Supabase."""
    global _engines
    if getattr(app.state, "rebuilding", False):
        return {"error": "Rebuild already in progress."}

    app.state.rebuilding = True
    _engines = {}
    gc.collect()

    def do_rebuild():
        try:
            build_engines_sequentially()
        finally:
            app.state.rebuilding = False

    background_tasks.add_task(do_rebuild)
    return {"message": "Rebuild started in background.", "status": "rebuilding"}

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
    """Shows which documents are retrieved for a query.
    Call: GET /debug-retrieve?q=tell+me+about+ramses&lang=en"""
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
    """Shows raw Supabase data for an entity.
    Call: GET /debug-entity?name=Ramesses+II"""
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
