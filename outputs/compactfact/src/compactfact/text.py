from __future__ import annotations
import re
import numpy as np

ATTRIBUTES = ("service_state", "tariff", "ticket_status", "action", "agreement")
CATALOG_KEYS = dict(zip(ATTRIBUTES, ("symptoms", "tariffs", "statuses", "actions", "agreements")))


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^а-яa-z0-9+]+", " ", str(text).lower().replace("ё", "е"))).strip()


def turns_as_rows(call):
    turns = call["turns"]
    rows = []
    for i, turn in enumerate(turns):
        prev = turns[i - 1] if i else {"text": "", "speaker": ""}
        nxt = turns[i + 1] if i + 1 < len(turns) else {"text": "", "speaker": ""}
        rows.append({"text": turn["text"], "prev": prev["text"], "next": nxt["text"],
                     "speaker": turn["speaker"], "prev_speaker": prev["speaker"],
                     "next_speaker": nxt["speaker"], "position": i / max(len(turns) - 1, 1),
                     "turn_idx": turn["idx"]})
    return rows


def context_text(row):
    return " ".join((row["prev"], row["text"], row["next"]))


def flags(row):
    """Observable discourse features; no annotation or call/account identifiers."""
    t, prev, nxt = (norm(row[k]) for k in ("text", "prev", "next"))
    patterns = [r"раньше|до этого|был[ао]? |работал|прежн|стар[ыо]", r"сейчас|теперь|сегодня|уже",
                r"предлож|можем|могу |можно|давайте", r"не помог|без результат|все равно|все ещ",
                r"не польз|не использ|отказ|отключ", r"вчера|назад|недел|месяц|с понедель|со втор|со сред|с четвер|с пят|с суб|с воскр",
                r"завтра|через |после двух", r"переш|смен|вместо|поменял", r"заявк|обращен|статус",
                r"перезагруз|провер|пробовал|сброс|замен|обнов|сделал", r"не |нет|ни "]
    vals = [float(row["speaker"] == "operator"), row["position"], min(len(t), 500) / 500,
            float("?" in row["text"]), float(row["next_speaker"] == "client"),
            float(bool(re.search(r"^(да|ага|хорош|ладно)|подходит|соглас|удобно|подтверж", nxt))),
            float(bool(re.search(r"да нет|не надо|не нуж|не знаю|подума|не подходит|не соглас", nxt))),
            float(bool(re.search(r"что пробовал|что делал|что предприним|какие действия", prev)))]
    vals += [float(bool(re.search(p, t))) for p in patterns]
    return np.asarray(vals, dtype=np.float32)