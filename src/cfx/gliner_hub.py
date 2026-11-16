"""Загрузка базовых моделей GLiNER с HF и выбор устройства (CUDA / Apple MPS / CPU)."""
import json, os, shutil


def device():
    import torch
    if os.environ.get("CFX_DEVICE"):          # принудительно: CFX_DEVICE=cpu, если на Mac GPU (MPS) что-то не работает
        return torch.device(os.environ["CFX_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_base(base):
    """Локальная копия с words_splitter_type=whitespace (в конфиге x-small/x-base — stanza);
    gliner-x-base на HF без файлов токенизатора → токенизатор mT5 из x-small (тот же словарь)."""
    if os.path.isdir(base):
        d = base
    else:
        from huggingface_hub import snapshot_download
        d = snapshot_download(base, local_dir="hf/" + base.split("/")[-1],
                              ignore_patterns=["*.bin", "*.fp16.*", "*.bf16.*", "onnx/*", "*.png"])
        if "gliner-x-base" in base or "gliner-x-large" in base:
            t = snapshot_download("knowledgator/gliner-x-small", local_dir="hf/gliner-x-small-tok",
                                  allow_patterns=["spiece.model", "tokenizer.json", "tokenizer_config.json",
                                                  "special_tokens_map.json", "added_tokens.json"])
            for f in ("spiece.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
                shutil.copy(os.path.join(t, f), os.path.join(d, f))
    cfg = os.path.join(d, "gliner_config.json")
    c = json.load(open(cfg))
    if c.get("words_splitter_type") != "whitespace":
        c["words_splitter_type"] = "whitespace"
        json.dump(c, open(cfg, "w"), indent=1)
    return d


REF_MS = 2330.0   # эталон: x-small (8 слоёв, ctx 25, fp32, 2 потока), golden[:4], мс на звонок


def calibrate():
    """Во сколько раз этот CPU медленнее эталонного: scale = REF / замер (для --lat_scale auto)."""
    import time, torch
    from gliner import GLiNER
    from gliner.data_processing.tokenizer import WordsSplitter
    from .data import splits
    from .gliner_fusion import GlinerSpans
    m = GLiNER.from_pretrained(resolve_base("knowledgator/gliner-x-small")).eval()
    m.data_processor.words_splitter = WordsSplitter("whitespace")
    torch.set_num_threads(2)
    calls = splits()["golden"][:4]
    gl = GlinerSpans(m, context_words=25, raw_threshold=0.1, batch=16)
    with torch.no_grad():
        gl.spans(calls[0]); t = time.time()
        for c in calls:
            gl.spans(c)
    ms = 1000 * (time.time() - t) / len(calls)
    return REF_MS / ms, ms
