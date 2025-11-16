# agent.py

import os
import json
import requests
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

# ========== CONFIG ==========
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========== LLM & Embeddings ==========
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

# ========== Calculator ==========
import numexpr as ne
from pydantic import BaseModel, Field


class CalcInput(BaseModel):
    expression: str = Field(
        description="Expression mathématique à évaluer, ex: '3*(2+5)**2'"
    )


@tool("calculator", args_schema=CalcInput)
def calculator(expression: str) -> str:
    """Calculette via numexpr pour évaluer une expression mathématique."""
    try:
        res = ne.evaluate(expression)
        return str(res.item() if hasattr(res, "item") else res)
    except Exception as e:
        return f"CALC_ERROR: {e}"


# ========== Tavily Search ==========
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

# ========== RAG (Chroma) ==========
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
vectorstore = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

RAG_INITIALIZED = False


def download(url: str) -> str:
    """Télécharge un fichier depuis une URL et le stocke dans ./downloaded_docs."""
    os.makedirs("./downloaded_docs", exist_ok=True)
    path = "./downloaded_docs/" + url.split("/")[-1]
    r = requests.get(url)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print("Downloaded", path)
    return path


def ingest_file(path: str) -> int:
    """Ingestion d’un fichier (PDF/CSV/TXT) dans le vector store Chroma."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)

    if path.lower().endswith(".pdf"):
        docs = PyPDFLoader(path).load()
    elif path.lower().endswith(".csv"):
        docs = CSVLoader(path).load()
    else:
        docs = TextLoader(path, encoding="utf-8").load()

    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {path}")
    return len(chunks)


def init_rag():
    """Initialise le RAG (télécharge + ingère le PDF) une seule fois."""
    global RAG_INITIALIZED
    if RAG_INITIALIZED:
        return
    url = (
        "https://raw.githubusercontent.com/Projet-MLOps-Team/Projet_MLOps-GenAI/main/conditions-tarifaires-particuliers-2025.pdf"
    )
    path = download(url)
    ingest_file(path)
    RAG_INITIALIZED = True
    print("✅ RAG initialisé (PDF conditions tarifaires ingéré)")


class RagInput(BaseModel):
    query: str = Field(description="Question en langage naturel.")
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre maximum de passages RAG à renvoyer.",
    )


@tool("rag_search", args_schema=RagInput)
def rag_search(query: str, k: int = 5) -> str:
    """Recherche des passages pertinents dans le vector store Chroma (RAG)."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return f"RAG_EMPTY: Aucun document trouvé pour la requête: {query}"

        docs = docs[:k]
        lines = [f"RAG_HITS: {len(docs)} résultats pour: {query}"]
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("file_path") or "unknown"
            page = meta.get("page", "?")
            txt = d.page_content.replace("\n", " ")
            if len(txt) > 600:
                txt = txt[:600] + "…"
            lines.append(f"[{i}] (page {page}) {src}: {txt}")
        return "\n".join(lines)
    except Exception as e:
        return f"RAG_ERROR: {e}"


# ========== ML Prediction Tool (remote .pkl on S3) ==========
import pandas as pd
import joblib
from io import BytesIO


class MLPredictInput(BaseModel):
    payload: Dict[str, Any] = Field(
        description="Dictionnaire de features pour la prédiction ML."
    )


MODEL_URL = "https://mlopsgenaiapp.s3.eu-west-3.amazonaws.com/best_model.pkl"
remote_model = None


def load_remote_model(url: str):
    """Télécharge un modèle pickle distant et le charge en mémoire."""
    print(f"📡 Téléchargement du modèle distant : {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    buffer = BytesIO(resp.content)
    model = joblib.load(buffer)
    print("✅ Modèle distant chargé en mémoire")
    return model


try:
    remote_model = load_remote_model(MODEL_URL)
except Exception as e:
    print(f"❌ ERREUR chargement modèle distant : {e}")
    remote_model = None


def _align_features(df: pd.DataFrame):
    """Aligne l'ordre et le set de features avec ceux utilisés au fit."""
    feature_names = getattr(remote_model, "feature_names_in_", None)
    if feature_names is None:
        return df

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(
            f"Features manquantes pour le modèle : {missing}. "
            f"Features reçues : {list(df.columns)}"
        )

    return df[list(feature_names)]


def _predict_remote(features: Dict[str, Any]) -> Dict[str, Any]:
    """Prédiction via modèle .pkl chargé depuis S3, avec sortie enrichie."""
    if remote_model is None:
        raise RuntimeError("Modèle distant non chargé.")

    df = pd.DataFrame([features])
    df = _align_features(df)

    y_pred = remote_model.predict(df)[0]

    proba_default = None
    if hasattr(remote_model, "predict_proba"):
        proba_default = float(remote_model.predict_proba(df)[0, 1])

    if int(y_pred) == 1:
        label_name = "Défaut probable"
    else:
        label_name = "Client plutôt sain"

    risk_level = None
    if proba_default is not None:
        if proba_default < 0.20:
            risk_level = "faible"
        elif proba_default < 0.50:
            risk_level = "modéré"
        else:
            risk_level = "élevé"

    if proba_default is not None and risk_level is not None:
        explanation = (
            f"Le modèle estime une probabilité de défaut d’environ "
            f"{proba_default*100:.1f} %, ce qui correspond à un risque {risk_level}."
        )
    else:
        explanation = (
            "Le modèle ne fournit pas de probabilité explicite, seulement une classe prédite."
        )

    return {
        "label": int(y_pred),
        "label_name": label_name,
        "proba_default": proba_default,
        "risk_level": risk_level,
        "explanation": explanation,
        "features_used": list(df.columns),
    }


def _jsonable(x: Any) -> Any:
    """Conversion best-effort en objet JSON-serialisable."""
    try:
        json.dumps(x)
        return x
    except TypeError:
        if hasattr(x, "tolist"):
            return x.tolist()
        return str(x)


@tool("ml_predict", args_schema=MLPredictInput)
def ml_predict(payload: Dict[str, Any]) -> str:
    """Effectue une prédiction via un modèle .pkl hébergé sur S3, avec sortie enrichie."""
    try:
        result = _predict_remote(payload)
        pretty = {
            "kind": "remote_pickle",
            "prediction": _jsonable(result),
        }
        return json.dumps(pretty, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"ML_ERROR: {e}"


# ========== SYSTEM PROMPT (texte) ==========
SYSTEM_PROMPT_TEXT = """
Tu es Sorbonnio, le chatbot bancaire de Kamila Kare, crée par Alexandre, Jiwon, Wai et Patricia. Tu es un assistant bancaire expert en défaut de crédit et conditions tarifaires 2025, doté d’une mémoire contextuelle
et de plusieurs outils spécialisés. Ton rôle est de sélectionner automatiquement l’outil pertinent,
d'utiliser intelligemment la mémoire issue du RAG, et de produire une réponse synthétique, fiable et systématique. 

[ MÉMOIRE ]
- Considère le contenu indexé dans le RAG comme ta mémoire fiable pour les tarifs bancaires.
- Consulte systématiquement `rag_search` pour toute requête liée à : tarifs, frais, commissions, comptes, cartes,
  packages, virements, incidents, clientèle (résident / non résident, jeune, premium, etc.).
- Ne JAMAIS inventer de montant : si les documents ne contiennent pas l’information, dis-le explicitement.

[ CHOIX DES OUTILS ]
1) RAG (`rag_search`) – PRIORITAIRE :
   - Utilise-le quand la question concerne des tarifs, frais, conditions, offres, segments de clientèle.
   - Formule une requête courte, précise, en français (ex: “tenue de compte actif non résident”).

2) Web Search (`web_search_tool`) :
   - Utilise-le pour les actualités, contexte macro, informations externes non présentes dans les documents.
   - Ne pas l’utiliser pour confirmer un chiffre qui devrait venir du PDF.

3) ML Prediction (`ml_predict`) :
   - Utilise-le si l’utilisateur demande une estimation de risque crédit ou une prédiction à partir de features.
   - Transmets fidèlement les features fournies et explique le résultat (classe, probabilité, niveau de risque).

4) Calculator (`calculator`) :
   - Utilise-le pour les calculs mathématiques explicites (montants, pourcentages, ratios).

[ COMPORTEMENT ]
- Si la question peut utiliser plusieurs outils, privilégie d’abord `rag_search`.
- Si `rag_search` renvoie RAG_EMPTY ou RAG_ERROR, explique que l’info n’est pas dans les documents et n’invente rien.
- Si aucun outil n’est pertinent, demande une clarification courte ou réponds avec ce que tu peux déduire sans halluciner.

[ STYLE ]
- Toujours en français.
- Réponses claires, concises, structurées.
- Tu termines toujours ta réponse en signant en bas de réponse: Sorbonnio, le chatbot bancaire de Kamila Kare
- Pour les tarifs, privilégie un tableau (type de compte | client | montant | périodicité) + une courte synthèse.
""".strip()


# ========== Agent factory ==========
def build_agent():
    """Construit l’agent ReAct avec les tools calcul, RAG, web et ML."""
    init_rag()

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
    tools = [calculator, rag_search, web_search_tool]
    if remote_model is not None:
        tools.append(ml_predict)

    # Prompt compatible avec create_react_agent (version récente) :
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TEXT),
            MessagesPlaceholder("messages"),
        ]
    )

    return create_react_agent(
        llm,
        tools,
        prompt=prompt,
    )


def chat(agent, messages: list, recursion_limit: int = 40) -> str:
    """
    messages = liste de dicts {"role": "user"/"assistant", "content": "..."}
    On convertit au format attendu par LangGraph: [("user", "..."), ("assistant", "..."), ...]
    """
    try:
        lc_messages = [(m["role"], m["content"]) for m in messages]
        out = agent.invoke(
            {"messages": lc_messages},
            config={"recursion_limit": recursion_limit},
        )
        return out["messages"][-1].content
    except Exception as e:
        return f"AGENT_ERROR: {e}"



# ========== MAIN ==========
if __name__ == "__main__":
    print("Bootstrapping agent...")

    agent = build_agent()

    print("\n[Calc]")
    print(chat(agent, "Calcule 3*(2+5)**2 et explique en une ligne."))

    print("\n[RAG]")
    print(
        chat(
            agent,
            "Résume-moi les frais de tenue de compte pour un non résident en utilisant ton outil rag_search.",
        )
    )

    print("\n[ML]")
    print(
        chat(
            agent,
            "Appelle ml_predict avec "
            "{'credit_lines_outstanding': 5, 'loan_amt_outstanding': 15000, "
            "'total_debt_outstanding': 25000, 'income': 60000, 'years_employed': 10, "
            "'fico_score': 720, 'debt_ratio': 0.3} et explique le résultat.",
        )
    )
