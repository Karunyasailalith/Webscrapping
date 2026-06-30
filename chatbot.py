import json
import re
from typing import List, Tuple, Dict, Optional
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ------------------ 0. CONFIG ------------------

ARTICLE_PATH = "C:/Users/Karunya/Downloads/webscrapping/article.json"
QUESTIONS_PATH = "C:/Users/Karunya/Downloads/webscrapping/questions.json"

# Small, free instruct model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Very simple English stopword list (to ignore generic words like "what", "is", "the")
STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "for", "to", "and",
    "or", "with", "do", "does", "did", "who", "whom", "whose", "where", "when",
    "why", "how", "which", "about", "this", "that", "these", "those", "be",
    "was", "were", "am", "i", "you", "he", "she", "it", "we", "they", "as",
    "at", "by", "from", "into", "than", "too", "very", "can", "could", "would",
    "should", "will", "shall", "have", "has", "had"
}


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def get_content_words(text: str) -> set:
    """
    Return only 'meaningful' words: no stopwords, no very short tokens.
    This is used for matching & safety checks.
    """
    norm = normalize_text(text)
    words = norm.split()
    return {w for w in words if w not in STOPWORDS and len(w) > 3}


# ------------------ 1. LOAD ARTICLE KNOWLEDGE ------------------

with open(ARTICLE_PATH, "r", encoding="utf-8") as f:
    article_data = json.load(f)

# Convert {heading: [para1, para2, ...]} -> list of (heading, full_text)
sections: List[Tuple[str, str]] = []
for heading, paragraphs in article_data.items():
    full_text = " ".join(paragraphs)
    sections.append((heading, full_text))


def get_section_by_name(name: str) -> Optional[Tuple[str, str]]:
    for heading, text in sections:
        if heading == name:
            return heading, text
    return None


# ------------------ 2. LOAD QUESTIONS HELP GUIDE ------------------

def safe_load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[INFO] Optional file not found: {path}. Skipping.")
        return {}
    except Exception as e:
        print(f"[WARN] Error loading {path}: {e}")
        return {}


raw_questions_guide = safe_load_json(QUESTIONS_PATH)

# exact map: normalized_question -> list[section_name]
question_to_sections: Dict[str, List[str]] = {}
# list for fuzzy matching: (normalized_guide_question, sections)
guide_question_list: List[Tuple[str, List[str]]] = []

# questions.json is: { "SectionName": [ "q1", "q2", ... ], ... }
for section_name, questions in raw_questions_guide.items():
    if not isinstance(questions, list):
        continue  # skip any malformed entries

    for q in questions:
        q_norm = normalize_text(q)

        # A question could theoretically belong to multiple sections
        if q_norm not in question_to_sections:
            question_to_sections[q_norm] = []
        if section_name not in question_to_sections[q_norm]:
            question_to_sections[q_norm].append(section_name)

        # For fuzzy matching, we store (normalized_question, [section_name])
        guide_question_list.append((q_norm, [section_name]))

print(f"[INFO] Loaded {len(question_to_sections)} guided questions.")


# ------------------ 3. SIMPLE RETRIEVAL (FALLBACK) ------------------

def score_section(question: str, text: str) -> float:
    """
    Simple similarity score using ONLY content words (no stopwords).
    """
    q_words = get_content_words(question)
    t_words = get_content_words(text)
    if not q_words or not t_words:
        return 0.0
    return len(q_words & t_words)


def get_top_sections(question: str, k: int = 3) -> List[Tuple[str, str]]:
    scored = []
    for heading, text in sections:
        s = score_section(question, text)
        if s > 0:
            scored.append((s, heading, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:k]
    return [(h, t) for s, h, t in top]


# ------------------ 4. FUZZY MATCHING AGAINST HELP-GUIDE ------------------

def fuzzy_match_guide(question: str, threshold: float = 0.5) -> Optional[List[str]]:
    """
    If user question is similar to any predefined guide question,
    return that guide question's sections.

    Similarity = overlap(content words) / len(user_question_content_words)
    """
    q_words = get_content_words(question)
    if not q_words:
        return None

    best_score = 0.0
    best_sections: Optional[List[str]] = None

    for guide_q_norm, sections_for_guide in guide_question_list:
        g_words = get_content_words(guide_q_norm)
        if not g_words:
            continue
        overlap = len(q_words & g_words)
        score = overlap / max(1, len(q_words))  # how much of user question is covered

        if score > best_score:
            best_score = score
            best_sections = sections_for_guide

    if best_score >= threshold:
        return best_sections

    return None


# ------------------ 5. LOAD LOCAL LLM MODEL (Qwen 0.5B, CPU-friendly) ------------------

print("Loading local LLM model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)
model.to(device)


# ------------------ 6. PROMPT BUILDING + GENERATION ------------------

def build_prompt(context_blocks: List[Tuple[str, str]], question: str) -> str:
    """
    Build a prompt that forces the model to answer ONLY from our JSON.
    """
    ctx_parts = []
    for heading, text in context_blocks:
        # truncate each section to keep input small and faster
        short_text = text[:1500]
        ctx_parts.append(f"### {heading}\n{short_text}")

    context_text = "\n\n".join(ctx_parts)

    prompt = (
        "You MUST answer ONLY using the CONTEXT below.\n"
        "If the answer is NOT present in the context, you MUST reply with EXACTLY:\n"
        "\"The article does not mention this.\"\n"
        "Do NOT add explanations, examples, guesses or outside knowledge.\n"
        "Do NOT try to answer from memory.\n"
        "Keep the answer short (1–3 sentences).\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer:"
    )

    return prompt


def generate_llm_answer(context_blocks: List[Tuple[str, str]], question: str) -> str:
    prompt = build_prompt(context_blocks, question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,   # shorter for speed and to avoid rambling
            do_sample=False       # greedy decoding
        )

    # keep only the part generated AFTER the prompt
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    answer = answer.strip()

    # 1) If our special sentence appears anywhere -> return exactly that
    lower_answer = answer.lower()
    if "the article does not mention this" in lower_answer or \
       "the article does not mention" in lower_answer:
        return "The article does not mention this."

    # 2) Strip leading "Answer:" if model repeats it
    if answer.lower().startswith("answer:"):
        answer = answer[len("answer:"):].lstrip()

    # 3) Cut at last full stop/question/exclamation to avoid half sentences
    last_dot = answer.rfind(".")
    last_q = answer.rfind("?")
    last_ex = answer.rfind("!")
    last_punct = max(last_dot, last_q, last_ex)

    if last_punct != -1:
        answer = answer[: last_punct + 1]

    # 4) Remove duplicate lines / extra whitespace
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    seen = set()
    deduped = []
    for line in lines:
        if line not in seen:
            deduped.append(line)
            seen.add(line)
    answer = " ".join(deduped).strip()

    # 5) Final safety rule:
    #    - If NO meaningful word from the question appears in the context -> "The article does not mention this."
    #    - Or if answer shares almost no content words with context -> also "The article does not mention this."
    context_text = " ".join(text[:1000] for _, text in context_blocks)

    q_content = get_content_words(question)
    ctx_content = get_content_words(context_text)
    ans_content = get_content_words(answer)

    # No overlap between question and context → clearly out of scope
    if q_content and ctx_content and not (q_content & ctx_content):
        return "The article does not mention this."

    # Very little overlap between answer and context → hallucination
    if ans_content and ctx_content:
        similarity = len(ans_content & ctx_content)
        if similarity < 2:
            return "The article does not mention this."

    return answer


# ------------------ 7. ANSWER PIPELINE ------------------

def answer_question_pipeline(question: str) -> Dict:
    """
    Pipeline:
    1) Exact match against help-guide questions (normalized)
    2) Fuzzy match against help-guide question variants
    3) Fallback: lexical retrieval over all sections
    """
    q_norm = normalize_text(question)

    # 1) Exact help-guide match
    if q_norm in question_to_sections:
        section_names = question_to_sections[q_norm]
        context_blocks: List[Tuple[str, str]] = []
        used_sections: List[str] = []

        for sec_name in section_names:
            sec = get_section_by_name(sec_name)
            if sec is not None:
                context_blocks.append(sec)
                used_sections.append(sec_name)

        if context_blocks:
            ans = generate_llm_answer(context_blocks, question)
            return {
                "answer": ans,
                "source_type": "guide_exact",
                "used_sections": used_sections,
            }

    # 2) Fuzzy help-guide match (any *related* question)
    fuzzy_sections = fuzzy_match_guide(question, threshold=0.5)
    if fuzzy_sections:
        context_blocks = []
        used_sections = []
        for sec_name in fuzzy_sections:
            sec = get_section_by_name(sec_name)
            if sec is not None:
                context_blocks.append(sec)
                used_sections.append(sec_name)

        if context_blocks:
            ans = generate_llm_answer(context_blocks, question)
            return {
                "answer": ans,
                "source_type": "guide_fuzzy",
                "used_sections": used_sections,
            }

    # 3) Fallback: simple retrieval over all sections
    top_sections = get_top_sections(question, k=3)

    if not top_sections:
        return {
            "answer": "The article does not mention this.",
            "source_type": "retrieval_none",
            "used_sections": [],
        }

    ans = generate_llm_answer(top_sections, question)
    used = [h for (h, _) in top_sections]

    return {
        "answer": ans,
        "source_type": "retrieval",
        "used_sections": used,
    }


# ------------------ 8. FASTAPI APP ------------------

app = FastAPI(title="AI Article RAG Chatbot (Help Guide + Qwen)")

# CORS (useful if you call this from a frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# simple health-check route
@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG chatbot is running"}


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    source_type: str
    used_sections: List[str]


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    result = answer_question_pipeline(payload.question)
    return AnswerResponse(**result)


# ------------------ 9. UVICORN ENTRYPOINT ------------------

# Run with:
#   uvicorn chatbot:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)

