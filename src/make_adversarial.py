"""Adversarial-подмножество: val с усиленным ASR-шумом (детерминированно, seed).

Модель шума выучена по парам «дословная цитата эталона ↔ текст реплики» в train (synth/generator.SUBS):
  * замены букв по матрице путаниц ASR (о→а, е→и, д→т, в→ф, …) в словах ≥ 4 букв с вероятностью level;
  * выпадение слов (p = level/10) и вставка слов-паразитов «эээ/ну/короче» (p = level/10);
  * у половины реплик — нижний регистр без пунктуации (как в сыром выводе ASR).
Эталонные факты не меняются: проверяется, что модель находит те же факты в более грязном тексте.

    python src/make_adversarial.py --level 0.3 --out data/val_adv_0.3.jsonl
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth.generator import SUBS  # noqa: E402

TOT = sum(n for _, _, n in SUBS)
FILLERS = ["эээ", "ну", "короче", "вот", "как бы"]


def sub_letters(w, rng, level):
    if len(w) < 4 or rng.random() > level:
        return w
    ch = list(w)
    for _ in range(1 + (rng.random() < level)):
        x = rng.random() * TOT
        for a, b, n in SUBS:
            x -= n
            if x <= 0:
                idx = [i for i, c in enumerate(ch) if c == a]
                if idx:
                    ch[rng.choice(idx)] = b
                break
    return "".join(ch)


def noisy(text, rng, level):
    out = []
    for w in text.split():
        r = rng.random()
        if r < level / 10:
            continue                                   # слово «съедено»
        if r < level / 5:
            out.append(rng.choice(FILLERS))            # паразит
        out.append(sub_letters(w, rng, level))
    s = " ".join(out) or text
    if rng.random() < 0.5:
        s = re.sub(r"[^\w\s-]", "", s.lower())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/val.jsonl")
    ap.add_argument("--level", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/val_adv_0.3.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for line in open(a.src, encoding="utf-8"):
            c = json.loads(line)
            for t in c["turns"]:
                t["text"] = noisy(t["text"], rng, a.level)
            for f in c.get("facts", []):              # спаны эталона после шума неточны → убираем
                f.get("source", {}).update({"span": None})
            fo.write(json.dumps(c, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} звонков → {a.out} (level={a.level}, seed={a.seed})")


if __name__ == "__main__":
    main()
