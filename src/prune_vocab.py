"""Урезание словаря GLiNER (mT5, 250k токенов) до русского + латиницы корпуса.

Что остаётся в словаре:
  * спецтокены (<pad>, </s>, <unk>) и токены GLiNER (<<ENT>>, <<SEP>>);
  * все кусочки на кириллице (+ ASCII внутри них);
  * знаки препинания, цифры, символы (без букв других алфавитов);
  * ASCII-кусочки, встречающиеся при токенизации корпуса (train/val/синтетика/каталог), и все ASCII длиной ≤ 2.
Остальное (китайский, арабский, хинди, редкие латинские подслова чужих языков…) удаляется.
Сегментация русского текста не меняется: проверяется на корпусе (--check).

Запуск: python src/prune_vocab.py --src models/g13 --dst models/g13_ru --corpus data/train.jsonl data/val.jsonl
"""
import argparse, json, os, re, shutil, unicodedata
import torch

CYR = re.compile("[Ѐ-ӿ]")
SP = "▁"


def piece_kind(p):
    s = p.replace(SP, "")
    if s == "":
        return "space"
    if all(ord(ch) < 128 for ch in s):
        return "ascii"
    if all(CYR.match(ch) or ord(ch) < 128 for ch in s):
        return "cyr"
    if all(not unicodedata.category(ch).startswith("L") for ch in s):
        return "symbol"
    return "other"


def corpus_texts(paths):
    for p in paths:
        for line in open(p, encoding="utf-8"):
            c = json.loads(line)
            for t in c.get("turns", []):
                yield t["text"]
            for f in c.get("facts", []):
                yield f.get("value", "")
                yield (f.get("object") or {}).get("name", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--catalog", default="data/catalog.json")
    ap.add_argument("--fp16", action="store_true", help="хранить веса в fp16 (вычисления всё равно fp32)")
    a = ap.parse_args()
    from tokenizers import Tokenizer

    tj = json.load(open(os.path.join(a.src, "tokenizer.json"), encoding="utf-8"))
    vocab = tj["model"]["vocab"]                      # [[piece, score], ...], id = позиция
    tok = Tokenizer.from_file(os.path.join(a.src, "tokenizer.json"))
    texts = list(corpus_texts(a.corpus))
    if os.path.exists(a.catalog):
        texts += [v for vals in json.load(open(a.catalog, encoding="utf-8")).values() for v in
                  (vals if isinstance(vals, list) else [])]
    used = set()
    for i in range(0, len(texts), 2048):
        for enc in tok.encode_batch(texts[i:i + 2048], add_special_tokens=False):
            used.update(enc.ids)

    keep = []
    for i, (p, _s) in enumerate(vocab):
        k = piece_kind(p)
        if i < 3 or k in ("space", "cyr", "symbol") or i in used or (k == "ascii" and len(p.replace(SP, "")) <= 2):
            keep.append(i)
    added = [t for t in tj["added_tokens"] if t["id"] >= len(vocab) and not t["content"].startswith(SP + "<extra_id")]
    old2new = {o: n for n, o in enumerate(keep)}
    for t in added:
        old2new[t["id"]] = len(old2new)
    K = len(old2new)
    print(f"словарь: {len(vocab) + sum(t['id'] >= len(vocab) for t in tj['added_tokens'])} -> {K} токенов; в корпусе использовано {len(used)}")

    # --- токенизатор
    tj2 = json.loads(json.dumps(tj))
    tj2["model"]["vocab"] = [vocab[i] for i in keep]
    tj2["model"]["unk_id"] = old2new[tj["model"]["unk_id"]]
    tj2["added_tokens"] = [{**t, "id": old2new[t["id"]]} for t in tj["added_tokens"] if t["id"] in old2new]
    for sect in ("post_processor",):
        s = json.dumps(tj2.get(sect))
        if s and '"id"' in s:   # T5: </s> id=1 не меняется (i<3), остальное не трогаем
            pass
    os.makedirs(a.dst, exist_ok=True)
    json.dump(tj2, open(os.path.join(a.dst, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    tc = json.load(open(os.path.join(a.src, "tokenizer_config.json"), encoding="utf-8"))
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {str(old2new[int(k)]): v for k, v in tc["added_tokens_decoder"].items()
                                      if int(k) in old2new}
    tc["extra_ids"] = 0                      # иначе T5Tokenizer допишет 100 токенов <extra_id_*>
    for key in ("additional_special_tokens", "extra_special_tokens"):
        if isinstance(tc.get(key), list):
            tc[key] = [x for x in tc[key] if "extra_id" not in str(x)]
    tc["backend"] = "tokenizers"
    json.dump(tc, open(os.path.join(a.dst, "tokenizer_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # --- конфиг
    cfg = json.load(open(os.path.join(a.src, "gliner_config.json"), encoding="utf-8"))
    cfg["vocab_size"] = K
    cfg["class_token_index"] = old2new[cfg["class_token_index"]]
    if isinstance(cfg.get("encoder_config"), dict):
        cfg["encoder_config"]["vocab_size"] = K
    json.dump(cfg, open(os.path.join(a.dst, "gliner_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # --- веса
    if os.path.exists(os.path.join(a.src, "pytorch_model.bin")):
        sd = torch.load(os.path.join(a.src, "pytorch_model.bin"), map_location="cpu")
    else:   # базовая модель с HF хранится в safetensors
        from safetensors.torch import load_file
        sd = load_file(os.path.join(a.src, "model.safetensors"))
    idx = torch.tensor(sorted(old2new, key=old2new.get))
    out, cache, V = {}, {}, cfg_vocab(sd)
    for k, v in sd.items():
        key = v.data_ptr()
        if key not in cache:            # связанные веса (shared = embed_tokens) остаются одним тензором
            w = v[idx].clone() if (v.dim() == 2 and v.shape[0] == V) else v.clone()
            cache[key] = w.half() if (a.fp16 and w.is_floating_point()) else w
        out[k] = cache[key]
    torch.save(out, os.path.join(a.dst, "pytorch_model.bin"))
    for f in ("arch.json", "train_log.json"):
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copy(os.path.join(a.src, f), a.dst)
    n_old = count(sd); n_new = count(out)
    meta = {"vocab_old": len(vocab) + sum(t["id"] >= len(vocab) for t in tj["added_tokens"]), "vocab_new": K,
            "params_old": n_old, "params_new": n_new, "fp16_storage": a.fp16}
    json.dump(meta, open(os.path.join(a.dst, "prune.json"), "w"), indent=1)
    print(f"параметры: {n_old/1e6:.1f}M -> {n_new/1e6:.1f}M")


def cfg_vocab(sd):
    return max(v.shape[0] for k, v in sd.items() if k.endswith("shared.weight"))


def count(sd):
    seen, n = set(), 0
    for v in sd.values():
        if v.data_ptr() in seen:
            continue
        seen.add(v.data_ptr()); n += v.numel()
    return n


if __name__ == "__main__":
    main()
