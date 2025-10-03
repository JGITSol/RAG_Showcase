from types import SimpleNamespace
from typing import List, Tuple, Optional


class FakeOllamaClientNumeric:
    """Fake Ollama client that returns numeric responses for evaluation/rerank prompts."""
    def __init__(self, host=None, numeric_response: str = '0.8', text_response: str = 'ok'):
        self.host = host
        self.numeric_response = numeric_response
        self.text_response = text_response

    def list(self):
        return []

    def generate(self, **kwargs):
        prompt = kwargs.get('prompt', '')
        if any(tok in prompt for tok in ['Relevance score', 'Rate', 'Rate relevance', 'Relevance score (0.0-1.0)']):
            return {'response': self.numeric_response}
        # default
        return {'response': self.text_response}


class FakeOllamaClientText:
    """Fake Ollama client that returns text responses for generation/fusion."""
    def __init__(self, host=None, response: str = 'fake answer'):
        self.host = host
        self._response = response

    def list(self):
        return []

    def generate(self, **kwargs):
        return {'response': self._response}


class FakeEmbeddings:
    def __init__(self, **kwargs):
        pass


class FakeChroma:
    def __init__(self, docs: Optional[List[SimpleNamespace]] = None, **kwargs):
        # accept kwargs like persist_directory, embedding_function, collection_name
        self._docs = docs or []

    def add_documents(self, docs: List[SimpleNamespace]):
        self._docs.extend(docs)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[SimpleNamespace, float]]:
        # Return up to k fake docs from stored docs, or generate simple ones
        if self._docs:
            out = []
            for i, d in enumerate(self._docs[:k]):
                # if stored as SimpleNamespace with page_content
                out.append((d, 0.6 + 0.1 * (i % 3)))
            return out

        # fallback synthetic docs
        return [
            (SimpleNamespace(page_content=f"Doc about {query} #{i}", metadata={'source': f'doc_{i}'}), 0.5 + 0.1 * i)
            for i in range(min(k, 3))
        ]


class EmptyChroma(FakeChroma):
    def __init__(self, **kwargs):
        super().__init__(docs=[])

    def similarity_search_with_score(self, query: str, k: int = 5):
        return []


def load_module_from_path(path: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
