"""Нормализация текста, строки-реплики и дискурсивные признаки (ловушки)."""
from __future__ import annotations

import hashlib
import re

import numpy as np

ATTRIBUTES = ("service_state", "tariff", "ticket_status", "action", "agreement")
SINGLE_VALUED = {"service_state", "tariff", "ticket_status"}
CATALOG_KEYS = {"service_state": ("symptoms",), "tariff": ("tariffs",), "ticket_status": ("statuses",),
                "action": ("actions",), "agreement": ("agreements",)}
WORD_RE = re.compile(r"[а-яёa-z0-9+]+")


def norm(text: str) -> str:
    t = str(text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^а-яa-z0-9+]+", " ", t)).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(str(text or "").lower().replace("ё", "е"))


def stem(w: str) -> str:
    return w[:5] if len(w) > 5 else w


def account_of(call: dict) -> str:
    s = call.get("series") or {}
    return s.get("account_id") or call["call_id"].split(":")[0]


def split_of(call: dict, seed: int = 42) -> str:
    """Детерминированное разбиение по аккаунту: fit 70% / rank 15% / dev 15%."""
    h = int(hashlib.md5(f"{seed}:{account_of(call)}".encode()).hexdigest(), 16) % 100
    return "fit" if h < 70 else "rank" if h < 85 else "dev"


def inference_view(call: dict) -> dict:
    """Белый список входных полей: эталонные факты в модель не попадают."""
    return {"call_id": call["call_id"], "started_at": call["started_at"],
            "duration_sec": call.get("duration_sec"), "series": call.get("series"),
            "turns": [{"idx": t["idx"], "speaker": t["speaker"], "text": t["text"]} for t in call.get("turns", [])]}


# --- дискурсивные признаки: подтверждение/отказ/предложение/вопрос/история ---
RX = {
    "past": re.compile(r"раньше|до этого|был[аои]?\b|работал|прежн|стар[ыо]|до\s+(вчера|понедел|вторн|сред|четвер|пятниц|суббот|воскрес)"),
    "now": re.compile(r"сейчас|теперь|сегодня|уже|до сих пор"),
    "offer": re.compile(r"предлож|можем|могу\b|можно\b|давайте|хотите|рекоменд|подключ[иу]м"),
    "fail": re.compile(r"не помог|без результат|все равно|все еще|ничего не измен|без толку|толку ноль"),
    "stopuse": re.compile(r"не польз|не использ|отказал|отключил"),
    "pastdate": re.compile(r"вчера|назад|прошл|с понедел|со вторн|со сред|с четвер|с пятниц|с суббот|с воскрес|на днях"),
    "futdate": re.compile(r"завтра|через\s|после\s+(двух|трех|обеда)|следующ|в\s+(понедел|вторн|сред|четвер|пятниц|суббот|воскрес)"),
    "switch": re.compile(r"переш[её]л|перешл|смен|вместо|поменял|перевел"),
    "ticket": re.compile(r"заявк|обращени|статус"),
    "did": re.compile(r"перезагруж|перезагрузил|проверил|пробовал|сбросил|заменил|обновил|сделал|переподключ|пополнил|оплатил"),
    "neg": re.compile(r"\bне\b|\bнет\b|\bни\b"),
    "imper": re.compile(r"^(попробуйте|перезагрузите|проверьте|сделайте|сбросьте|отключите|включите|зайдите|нажмите)"),
    "commit": re.compile(r"фиксир|оформил|оформлю|записываю|записал|договорил|назначен|подтвержд|готово|поставлю|отправ"),
}
CONFIRM = re.compile(r"^(да\b|ага\b|угу\b|хорош|ладно|давайте|конечно|отлично)|подходит|соглас|подтвержда|договорились|устраивает|удобно")
REJECT = re.compile(r"^(да\s+нет|нет\b|не\s+надо)|не подходит|не соглас|не надо|не нуж|не хочу|не стоит|подума|откаж")


def turn_rows(call: dict) -> list[dict]:
    turns = call["turns"]
    rows = []
    n = len(turns)
    for i, t in enumerate(turns):
        prev = turns[i - 1] if i else {"text": "", "speaker": ""}
        nxt = turns[i + 1] if i + 1 < n else {"text": "", "speaker": ""}
        nxt2 = turns[i + 2] if i + 2 < n else {"text": "", "speaker": ""}
        rows.append({"i": i, "turn_idx": t["idx"], "text": t["text"], "speaker": t["speaker"],
                     "prev": prev["text"], "prev_speaker": prev["speaker"], "next": nxt["text"],
                     "next_speaker": nxt["speaker"], "next2": nxt2["text"], "pos": i / max(n - 1, 1)})
    return rows


FLAG_NAMES = ["is_operator", "pos", "len", "question", "next_client", "next_confirm", "next_reject",
              "next2_confirm", "prev_asks_tried", "prev_question", "prev_offer"] + [f"rx_{k}" for k in RX]


def flags(row: dict) -> np.ndarray:
    t, prev, nxt, nxt2 = norm(row["text"]), norm(row["prev"]), norm(row["next"]), norm(row["next2"])
    v = [float(row["speaker"] == "operator"), row["pos"], min(len(t), 400) / 400, float("?" in row["text"]),
         float(row["next_speaker"] == "client"), float(bool(CONFIRM.search(nxt))), float(bool(REJECT.search(nxt))),
         float(bool(CONFIRM.search(nxt2))),
         float(bool(re.search(r"что (вы )?(уже )?(пробовал|делал|предприн)|перезагружал|проверял", prev))),
         float("?" in row["prev"]), float(bool(RX["offer"].search(prev)))]
    v += [float(bool(rx.search(t))) for rx in RX.values()]
    return np.asarray(v, dtype=np.float32)
