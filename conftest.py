import sys
import types


def _make_module(name: str):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# Stub chromadb package to avoid importing the real native extension during tests
chromadb = _make_module("chromadb")
_make_module("chromadb.api")
_make_module("chromadb.api.client")
_make_module("chromadb.api.models")
_make_module("chromadb.utils")
_make_module("chromadb.utils.embedding_functions")

# Provide minimal client and embedding function objects used by code paths
try:
    from chromadb.api.client import Client as _RealClient  # type: ignore
except Exception:
    class Client:  # minimal placeholder
        def __init__(self, *args, **kwargs):
            pass

    # Attach placeholder to our stub modules
    sys.modules["chromadb.api.client"].Client = Client  # type: ignore

def _default_embedding_function():
    class DummyEF:
        def __call__(self, texts):
            # Return a list of fixed-dimension vectors (1-dim) to satisfy callers
            return [[0.0] for _ in texts]

    return DummyEF()

# Attach a DefaultEmbeddingFunction factory if code expects it
sys.modules["chromadb.utils.embedding_functions"].DefaultEmbeddingFunction = _default_embedding_function  # type: ignore


# Stub onnxruntime to avoid heavy native import errors during collection
onnxruntime = _make_module("onnxruntime")

class _DummyInferenceSession:
    def __init__(self, *args, **kwargs):
        pass

onnxruntime.InferenceSession = _DummyInferenceSession  # type: ignore


# Prevent other optional heavy packages from breaking imports (best-effort)
for name in ("onnxruntime.capi", "onnxruntime.capi._pybind_state"):
    if name not in sys.modules:
        _make_module(name)


# ---------------------------------------------------------------------------
# Lightweight stubs for langchain, ollama, sentence_transformers, networkx
# These let tests import modules and run without installing heavy dependencies
# ---------------------------------------------------------------------------

# langchain package and submodules
langchain = _make_module("langchain")
text_splitter_mod = _make_module("langchain.text_splitter")
docstore_mod = _make_module("langchain.docstore")
docstore_doc = _make_module("langchain.docstore.document")
emb_mod = _make_module("langchain.embeddings")
vectorstores_mod = _make_module("langchain.vectorstores")

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=50, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str):
        # Very small splitter: return full text as one chunk
        return [text]

docstore_mod.document = docstore_doc

class Document:
    def __init__(self, page_content: str = "", metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def OllamaEmbeddings(base_url: str = None, model: str = None):
    # Return a simple callable embedding function
    class _E:
        def __call__(self, texts):
            return [[0.0] for _ in texts]

    return _E()

class Chroma:
    def __init__(self, persist_directory: str = None, embedding_function=None, collection_name: str = None):
        self._persist = persist_directory
        self._embedding = embedding_function
        self._collection_name = collection_name
        self._docs = []

    def add_documents(self, doc_objects):
        self._docs.extend(doc_objects)

    def similarity_search_with_score(self, query: str, k: int = 5):
        # Return up to k dummy docs with descending scores
        res = []
        for i, doc in enumerate(self._docs[:k]):
            res.append((doc, 1.0 - (i * 0.1)))
        return res

    @property
    def _collection(self):
        class _C:
            def count(self_inner):
                return len(self._docs)

        return _C()

# Bind stubs into sys.modules
text_splitter_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter  # type: ignore
docstore_doc.Document = Document  # type: ignore
emb_mod.OllamaEmbeddings = OllamaEmbeddings  # type: ignore
vectorstores_mod.Chroma = Chroma  # type: ignore


# ollama client stub
ollama = _make_module("ollama")

class _DummyOllamaClient:
    def __init__(self, host: str = None):
        self.host = host

    def list(self):
        return []

    def generate(self, model: str = None, prompt: str = "", options: dict = None):
        # Simple heuristic: if prompt requests a numeric score, return '0.5'
        low = prompt.lower()
        if "relevance score" in low or "rate" in low or "score (0.0-1.0)" in low:
            return {"response": "0.5"}
        # If asking for a decision token, return '[Retrieve]'
        if "respond with exactly" in low and "[retrieve]" in low:
            return {"response": "[Retrieve]"}
        # Default: return a placeholder answer
        return {"response": "This is a stubbed response."}

ollama.Client = _DummyOllamaClient  # type: ignore


# sentence_transformers stub
_make_module("sentence_transformers")
sys.modules["sentence_transformers"].SentenceTransformer = lambda name=None: None  # type: ignore

# networkx stub
_make_module("networkx")

