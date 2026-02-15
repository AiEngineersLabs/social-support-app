import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings as LlamaSettings
from app.config import settings

_index = None


def _get_embedding_model():
    return OllamaEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def _get_llm():
    return Ollama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        request_timeout=120,
    )


def init_vector_store():
    global _index

    LlamaSettings.embed_model = _get_embedding_model()
    LlamaSettings.llm = _get_llm()

    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("social_support_policies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    return _index


def ingest_policy_documents(documents: list[dict]):
    """Ingest policy documents into the vector store.
    Each doc: {"text": "...", "metadata": {...}}
    """
    global _index

    LlamaSettings.embed_model = _get_embedding_model()
    LlamaSettings.llm = _get_llm()

    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("social_support_policies")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llama_docs = [
        Document(text=doc["text"], metadata=doc.get("metadata", {}))
        for doc in documents
    ]

    _index = VectorStoreIndex.from_documents(
        llama_docs,
        storage_context=storage_context,
    )
    return _index


def query_policies(query: str, top_k: int = 3) -> str:
    """Query the policy vector store for relevant information."""
    global _index
    if _index is None:
        init_vector_store()

    if _index is None:
        return "No policy documents have been ingested yet."

    query_engine = _index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return str(response)
