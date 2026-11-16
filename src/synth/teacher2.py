"""Учитель v2 (DeepSeek / любой OpenAI-совместимый сервер): план + скелет → живой ASR-транскрипт с цитатами фактов.

Шум распознавания добавляет сама LLM по образцу реальных транскриптов train (уровень — в запросе).
Проверка ответа кодом (validate2): цитата есть в реплике, по смыслу ближе всего к значению факта
(char-TF-IDF по банку значений модели), относительная дата прозвучала рядом (нечётко), ≥80% фактов на месте.
"""
from __future__ import annotations

import json, os, re, sys
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from teacher import _env, _esc  # noqa: E402

REAL_ASR = """operator: Паняла вас, это индикация активности, всё нормально.
client: Да, верно, всё так. Ну хоть один щиловек по-человечески объяснил. Ладно, спасиба.
operator: Ясно. А вы что-нибудь уже проповали? Перезагрузку, к примеру?
client: После трёх? Ну... да, подходит. Сын как раз после работы заедет, а то Марь Сергевна не всё опъяснила.
client: Да падходит спасибо И тариф у миня стати не менялся Я вчера у соседки видела у неё птю-то новенькое…
operator: Спасибо нашла вас Марин Викторовна да Расскажите что оминно случилось
operator: в жетвырг всё будет готово спасибо за обращение всего доброго!
client: Да тоже на прашлой недыле, на прошлой неделе я и увидела — индикатор не горит, и сижу, думаю, что же это такое.
operator: Да-да слушаю Просто хутыла сказать  у вас интирнету трафик небольшой ходит можем подключить пакет дополнительных гигабайтов выгодно выйдет"""

SYSTEM2 = """Ты пишешь реалистичные транскрипты звонков в техподдержку оператора связи на русском языке — в том виде,
в каком их выдаёт система распознавания речи (ASR). Примеры реальных транскриптов:
""" + REAL_ASR + """

Тебе дают: ДАТУ ЗВОНКА, ПЛАН ФАКТОВ (что должно прозвучать), СКЕЛЕТ звонка (по шагам), ЛОВУШКИ, УРОВЕНЬ ШУМА.
Правила:
1. Каждый шаг скелета превращай в 1–3 реплики живого разговора: «ну», «эээ», «вот», перебивки, самоисправления,
   связки («минутку», «секунду, проверяю»). Новых фактов не добавляй: никаких других устройств, тарифов, заявок, договорённостей.
2. Каждый факт плана должен прозвучать. Для каждого верни:
   turn_idx — реплика-источник факта (для договорённости — реплика, где клиент соглашается),
   quote_turn_idx и quote — реплика и ТОЧНАЯ подстрока из неё (2–12 слов), где названо САМО значение факта:
   что сломалось / какой тариф / какой статус / что клиент сделал / о чём договорились. НЕ дата и НЕ слова согласия
   («да, подходит») — для договорённости quote берётся из реплики оператора с предложением.
3. Относительные даты (date_expr в плане) произнеси в реплике факта или соседней; смысл не меняй
   («со вторника», «дня три уже как», «вчера с утра»). Конкретных календарных дат не называй.
4. Где в скелете «сказать своими словами» — не повторяй формулировку плана, перефразируй как в жизни.
5. Ловушки разыграй так, чтобы они НЕ стали фактами (отказ клиента, поправка, гипотеза, чужая ситуация).
6. УРОВЕНЬ ШУМА: лёгкий — редкие опечатки и пропущенная пунктуация; средний — замены похожих звуков (о/а, е/и, д/т, б/п, ч/щ),
   склейки и обрывы слов, повторы; сильный — частые искажения, оборванные реплики, «съеденные» окончания, неверно
   распознанные слова, но смысл фактов угадывается. quote копируй из уже зашумлённого текста — символ в символ.
7. Никаких реальных ФИО, телефонов, адресов. Имя оператора — только имя (и отчество).
Ответ — только JSON:
{"turns":[{"idx":0,"speaker":"operator","text":"..."}],"facts":[{"plan_id":0,"turn_idx":3,"quote_turn_idx":3,"quote":"..."}]}"""

USER2 = """ДАТА ЗВОНКА: {started_at}
УРОВЕНЬ ШУМА: {noise}
ПЛАН ФАКТОВ:
{plan}
СКЕЛЕТ:
{beats}
ЛОВУШКИ: {traps}
ДОПОЛНИТЕЛЬНО: {notes}"""


class Turn(BaseModel):
    idx: int
    speaker: str
    text: str


class FactRef(BaseModel):
    plan_id: int
    turn_idx: int
    quote_turn_idx: Optional[int] = None
    quote: str = ""


class Out2(BaseModel):
    turns: List[Turn]
    facts: List[FactRef]


def payload(c):
    plan = [{"plan_id": i, "attribute": f["attribute"], "value": f["value"], "object": f["object"]["name"],
             "speaker": f["speaker"], "date_expr": f["date_expr"], "step": f["beat"]} for i, f in enumerate(c["plan"])]
    return {"started_at": c["started_at"][:10], "noise": c["noise"], "plan": json.dumps(plan, ensure_ascii=False),
            "beats": "\n".join(f"{i}. [{b['speaker']}] {b['text']}" for i, b in enumerate(c["beats"])),
            "traps": ", ".join(c["traps"]) or "нет", "notes": "; ".join(c["notes"]) or "нет"}


def messages(c):
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([("system", _esc(SYSTEM2)), ("user", USER2)]).format_messages(**payload(c))


def parse(text):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = re.sub(r"```(?:json)?", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON")
    return Out2.model_validate_json(t[a:b + 1])


# ------------------------------------------------------------------ проверка
_N = lambda s: re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip(" .,!?…—-«»\"")


def _tri(s):
    s = " " + _N(s) + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _cont(small, big):
    a = _tri(small)
    return len(a & _tri(big)) / max(len(a), 1)


class Validator:
    def __init__(self, ex):
        from sklearn.preprocessing import normalize
        self.ex, self.normalize = ex, normalize

    def value_ok(self, attr, value, quote, paraphrase):
        if _cont(value, quote) >= 0.45:
            return True
        if value == "больше не используется" and re.search(r"(пере|не )ст.{0,4}пол|не польз|отказ|отключ", _N(quote)):
            return True
        bank = self.ex.banks.get(attr)
        if bank is None:
            return False
        from cfx.text import norm
        j = bank["index"].get(norm(value))
        if j is None:
            return False
        sims = (self.normalize(self.ex.vec.transform([quote])) @ bank["lex"].T).toarray()[0]
        rank = int((sims > sims[j]).sum())
        return rank < (12 if paraphrase else 5) and sims[j] >= 0.12

    def repair(self, turns, ti, q, value):
        from cfx.gliner_data import tokenize, find_window
        for j in dict.fromkeys([q, ti, ti - 1, ti - 2]):
            if not 0 <= j < len(turns):
                continue
            toks = tokenize(turns[j]["text"])
            w = find_window(toks, value)
            if w:
                return j, turns[j]["text"][toks[w[0]][1]:toks[w[1]][2]]
        return None

    def date_ok(self, f, cd, near):
        ex = f.get("date_expr")
        if not ex:
            return True
        if _cont(ex, near) >= 0.5:
            return True
        from cfx.dates_bank import extract
        want = datetime.fromisoformat(f["valid_from"]).date()
        have = {d for *_, ds in extract(near, cd) for d in ds}
        return any(abs((d - want).days) <= 1 for d in have)

    def __call__(self, c, out: Out2):
        turns = [t.model_dump() for t in out.turns]
        for i, t in enumerate(turns):
            t["idx"] = i
        cd = datetime.fromisoformat(c["started_at"]).date()
        para_beats = {i for i, b in enumerate(c["beats"]) if "своими словами" in b["text"]}
        facts, seen = [], set()
        self.why = {}
        for ref in out.facts:
            if not 0 <= ref.plan_id < len(c["plan"]) or ref.plan_id in seen or not 0 <= ref.turn_idx < len(turns):
                continue
            q = ref.quote_turn_idx if ref.quote_turn_idx is not None else ref.turn_idx
            if not 0 <= q < len(turns) or abs(q - ref.turn_idx) > 2 or not ref.quote.strip():
                continue
            f = dict(c["plan"][ref.plan_id])
            para = f["beat"] in para_beats
            good = _N(ref.quote) in _N(turns[q]["text"]) and self.value_ok(f["attribute"], f["value"], ref.quote, para)
            if not good:                          # починка: ищем само значение в реплике факта и двух предыдущих
                rq = self.repair(turns, ref.turn_idx, q, f["value"])
                if rq is None:
                    self.why[ref.plan_id] = f"значение: «{f['value']}» vs «{ref.quote}»"; continue
                q, ref.quote = rq
            lo, hi = max(0, min(q, ref.turn_idx) - 1), min(len(turns), max(q, ref.turn_idx) + 2)
            if not self.date_ok(f, cd, " ".join(turns[j]["text"] for j in range(lo, hi))):
                self.why[ref.plan_id] = f"дата «{f.get('date_expr')}» не найдена"; continue
            txt = turns[q]["text"]
            k = _N(txt).find(_N(ref.quote))
            seen.add(ref.plan_id)
            f["source"] = {"turn_idx": ref.turn_idx, "span": None, "quote": ref.quote}
            f["gl_quote"] = {"turn_idx": q, "text": ref.quote}
            facts.append({k2: v for k2, v in f.items() if k2 not in ("beat",)})
        if len(facts) < 0.8 * len(c["plan"]):
            return None, f"прошло {len(facts)}/{len(c['plan'])} фактов"
        return {"call_id": c["call_id"], "started_at": c["started_at"], "duration_sec": None, "series": None,
                "turns": turns, "facts": facts, "synthetic": True, "block": c["block"], "noise": c["noise"],
                "teacher": _env("TEACHER_MODEL", "deepseek-flash")}, "ok"
