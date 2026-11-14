"""Маркеры разговорной речи: частоты токенов-дизфлюенций и их позиций в репликах.

Источники: (1) реальные транскрипты телефонных звонков open_stt (HF: Joarnixon/open-stt-calls),
(2) обучающий корпус кейса (реплики клиента и оператора отдельно),
(3) повседневные диалоги (HF: kukunechka/russian-everyday-dialogues, Hacker1337/ru_dialogsum) — фон.
Результат — вероятностная модель: P(маркер в начале реплики), P(вставка внутри), распределение маркеров.
"""
import json, re, sys, collections
CAND = ["эээ", "ээ", "э", "ммм", "мм", "м", "ну", "вот", "короче", "блин", "типа", "как бы", "это самое", "значит",
        "слушайте", "слушай", "так", "а", "да", "ага", "угу", "ой", "ну вот", "ну да", "в общем", "в смысле",
        "капец", "честно говоря", "понимаете", "знаете", "кстати", "вообще", "просто", "уже", "ещё", "же", "ладно",
        "хорошо", "нет", "ну это", "секунду", "сейчас", "щас"]
TOK = re.compile(r"[а-яё]+")

def stats(texts):
    first, inner, n_turns, n_tok = collections.Counter(), collections.Counter(), 0, 0
    for t in texts:
        w = TOK.findall(t.lower())
        if not w: continue
        n_turns += 1; n_tok += len(w)
        for m in CAND:
            mw = m.split()
            L = len(mw)
            for i in range(len(w) - L + 1):
                if w[i:i + L] == mw:
                    (first if i == 0 else inner)[m] += 1
    return {"turns": n_turns, "tokens": n_tok,
            "p_first": {m: round(first[m] / n_turns, 5) for m in CAND if first[m]},
            "p_inner_per_token": {m: round(inner[m] / n_tok, 6) for m in CAND if inner[m]},
            "dash_cut": round(sum(t.rstrip().endswith(("—", "-")) for t in texts) / max(1, len(texts)), 4),
            "ellipsis": round(sum("..." in t or "…" in t for t in texts) / max(1, len(texts)), 4)}

if __name__ == "__main__":
    out = {}
    calls = [json.loads(l) for l in open("data/train.jsonl")]
    out["case_client"] = stats([t["text"] for c in calls for t in c["turns"] if t["speaker"] == "client"])
    out["case_operator"] = stats([t["text"] for c in calls for t in c["turns"] if t["speaker"] == "operator"])
    out["openstt_calls"] = stats([json.loads(l)["sentence"] or "" for l in open("../ext/openstt_calls.jsonl")])
    try:
        import pandas as pd
        df = pd.read_parquet("../ext/ru_dialogsum_train.parquet")
        col = [c for c in df.columns if "dialog" in c.lower()][0]
        out["ru_dialogsum"] = stats([x for d in df[col].head(3000) for x in str(d).split("\n")])
    except Exception as e:
        out["ru_dialogsum"] = {"error": str(e)[:200]}
    json.dump(out, open("runs/markers.json", "w"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        if "p_first" in v:
            top = sorted(v["p_first"].items(), key=lambda x: -x[1])[:8]
            inner = sorted(v["p_inner_per_token"].items(), key=lambda x: -x[1])[:8]
            print(k, v["turns"], "turns | first:", top, "| inner:", inner, "| cut", v["dash_cut"])
        else:
            print(k, v)
