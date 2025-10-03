import sys
import types

# Provide lightweight stubs for heavy libraries to avoid importing native extensions during tests
heavy_modules = [
    "torch",
    "tensorflow",
    "transformers",
    "sentence_transformers",
    "opentelemetry",
    "opentelemetry.sdk",
    "opentelemetry.sdk.trace",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.sdk.metrics",
    "opentelemetry.sdk.metrics._internal",
    "sentence_transformers.evaluation",
    "sentence_transformers.cross_encoder",
]

for name in heavy_modules:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# Minimal attributes used by the code
if not hasattr(sys.modules["sentence_transformers"], "SentenceTransformer"):
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
    sys.modules["sentence_transformers"].SentenceTransformer = SentenceTransformer

# Provide a minimal torch namespace to avoid attribute errors
if not hasattr(sys.modules["torch"], "nn"):
    torch_mod = sys.modules["torch"]
    torch_mod.nn = types.SimpleNamespace()
    torch_mod.device = lambda *a, **k: None
    torch_mod.tensor = lambda *a, **k: None

# Minimal transformers namespace
if not hasattr(sys.modules["transformers"], "PreTrainedModel"):
    sys.modules["transformers"].PreTrainedModel = object

# Minimal opentelemetry objects used by some libs
otel = sys.modules.get("opentelemetry")
if otel is not None:
    otel.sdk = types.SimpleNamespace()
    otel.sdk.trace = types.SimpleNamespace()
    otel.sdk.metrics = types.SimpleNamespace()

# Ensure langchain submodules stubs exist (tests also create own stubs)
if "langchain" not in sys.modules:
    sys.modules["langchain"] = types.ModuleType("langchain")

# silence potential warnings from pytest-asyncio configuration
import pytest
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio")
