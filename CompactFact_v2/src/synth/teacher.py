"""LangChain-цепочки: учитель-LLM для дистилляции, верификатор спорных фактов, экстрактор как Runnable.

1) DistillChain: план фактов + черновик диалога (generator.py) → естественный диалог с теми же фактами.
   Учитель (>17B, DeepSeek через OpenAI-совместимый API) переписывает реплики, сохраняя факты и
   относительные выражения дат; ответ разбирается в Pydantic-схему и валидируется кодом:
   значение факта должно найтись в реплике-источнике (нечётко), выражение даты — в банке.
2) VerifierChain: «выборочная проверка спорных фактов крупной моделью» (критерий 5 кейса) —
   только факты с уверенностью в серой зоне [lo, hi] уходят в LLM; остальное решает компактная модель.
3) extractor_runnable: экстрактор как LangChain Runnable — встраивается в агентные цепочки и
   конвейеры памяти (Graphiti/Zep) без LLM-вызова на извлечение.
Учитель — любой OpenAI-совместимый сервер (vLLM / Ollama / LM Studio / DeepSeek): TEACHER_BASE_URL,
TEACHER_MODEL, TEACHER_API_KEY (или DEEPSEEK_API_KEY). Массовый прогон — src/synth/run_distill.py.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from pydantic import BaseModel, Field

SYSTEM_DISTILL = """Ты — генератор обучающих данных для системы извлечения фактов из звонков в техподдержку оператора связи.
Тебе дают ПЛАН ФАКТОВ (JSON) и ЧЕРНОВИК диалога. Перепиши диалог так, чтобы он звучал как живой телефонный
разговор на русском: разговорные обороты, перебивания, самоисправления, маркеры речи с частотами из СТИЛЯ.
Строгие правила:
1. Каждый факт плана должен быть озвучен в реплике с указанным speaker; верни его turn_idx.
2. Относительные выражения дат («вчера», «со вторника», «через три дня», «в четверг после двух») сохраняй дословно.
3. Не добавляй новых фактов. Ловушки (предложение оператора без согласия клиента, отвергнутая догадка) — оставь как есть.
4. Не используй реальные ФИО, номера телефонов и адреса.
5. Ответ — строго JSON по схеме: {"turns":[{"idx":0,"speaker":"operator|client","text":"..."}],
   "facts":[{"plan_id":0,"turn_idx":3}]}"""

USER_DISTILL = """СТИЛЬ (частоты маркеров по реальным звонкам): {style}
ДАТА ЗВОНКА: {started_at}
ПЛАН ФАКТОВ:
{plan}
ЧЕРНОВИК:
{draft}"""

SYSTEM_VERIFY = """Ты проверяешь один извлечённый факт из звонка в техподдержку. Ответь JSON:
{"supported": true|false, "reason": "кратко"}. Факт подтверждён, только если он прямо следует из реплик,
а не является предложением оператора без согласия клиента, отвергнутой догадкой или вопросом."""

USER_VERIFY = """Фрагмент диалога:
{snippet}
Факт: атрибут={attribute}; значение={value}; объект={object}; говорящий={speaker}; действует с={valid_from}"""


class TurnOut(BaseModel):
    idx: int
    speaker: str
    text: str


class FactRef(BaseModel):
    plan_id: int
    turn_idx: int


class DistillOut(BaseModel):
    turns: List[TurnOut]
    facts: List[FactRef]


class VerifyOut(BaseModel):
    supported: bool
    reason: Optional[str] = Field(default=None)


def _env(name, default=None):
    v = os.environ.get(name)
    if v:
        return v
    for p in (".env", "../.env", os.path.expanduser("~/Desktop/MTS/.env")):
        if os.path.exists(p):
            for line in open(p):
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip()
    return default


def _key():
    return _env("TEACHER_API_KEY") or _env("DEEPSEEK_API_KEY")


def llm(model=None, temperature=0.7, base_url=None, api_key=None, json_mode=None, max_tokens=None):
    """Любой OpenAI-совместимый сервер: vLLM, Ollama, LM Studio, llama.cpp server, DeepSeek.
    Настройка через окружение: TEACHER_BASE_URL, TEACHER_MODEL, TEACHER_API_KEY, TEACHER_JSON_MODE."""
    from langchain_openai import ChatOpenAI
    base_url = base_url or _env("TEACHER_BASE_URL", "https://api.deepseek.com")
    model = model or _env("TEACHER_MODEL", "deepseek-chat")
    api_key = api_key or _key() or "local"          # локальным серверам ключ не нужен
    json_mode = (_env("TEACHER_JSON_MODE", "1") == "1") if json_mode is None else json_mode
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=temperature,
                      model_kwargs=kw, timeout=600, max_retries=2, max_tokens=max_tokens)


def _esc(t: str) -> str:
    """Системные промпты содержат JSON-примеры: экранируем фигурные скобки для шаблона LangChain."""
    return t.replace("{", "{{").replace("}", "}}")


def parse_json_obj(text: str, model_cls):
    """Разбор ответа локальной модели: убирает <think>…</think> и ```json, берёт первый {...}."""
    import re as _re
    t = _re.sub(r"<think>.*?</think>", "", text, flags=_re.S)
    t = _re.sub(r"```(?:json)?", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in response")
    return model_cls.model_validate_json(t[a:b + 1])


def distill_chain(**llm_kw):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    prompt = ChatPromptTemplate.from_messages([("system", _esc(SYSTEM_DISTILL)), ("user", USER_DISTILL)])
    return prompt | llm(**llm_kw) | StrOutputParser() | RunnableLambda(lambda t: parse_json_obj(t, DistillOut))


def verifier_chain(**llm_kw):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    prompt = ChatPromptTemplate.from_messages([("system", _esc(SYSTEM_VERIFY)), ("user", USER_VERIFY)])
    return prompt | llm(temperature=0.0, **llm_kw) | StrOutputParser() | RunnableLambda(lambda t: parse_json_obj(t, VerifyOut))


def plan_of(call):
    return [{"plan_id": i, "attribute": f["attribute"], "value": f["value"], "object": f["object"]["name"],
             "speaker": f["speaker"], "draft_turn": f["source"]["turn_idx"]} for i, f in enumerate(call["facts"])]


def validate(call, out: DistillOut):
    """Код-проверка ответа учителя: факт озвучен в реплике-источнике, даты восстанавливаются."""
    from cfx.objects import word_close
    from cfx.text import words
    turns = [t.model_dump() for t in out.turns]
    ok_facts = []
    for ref in out.facts:
        if not 0 <= ref.plan_id < len(call["facts"]) or not 0 <= ref.turn_idx < len(turns):
            continue
        f = dict(call["facts"][ref.plan_id])
        vw = [w for w in words(f["value"]) if len(w) >= 4]
        # значение должно прозвучать в реплике-источнике или соседней (согласие клиента идёт после предложения оператора)
        cov = 0.0
        for j in range(max(0, ref.turn_idx - 1), min(len(turns), ref.turn_idx + 2)):
            tw = words(turns[j]["text"])
            cov = max(cov, sum(any(word_close(w, t) for t in tw) for w in vw) / max(len(vw), 1))
        if cov < 0.6:
            continue
        # дата: если факт датирован не днём звонка, выражение даты должно остаться в тексте (реплика ±1)
        from datetime import date as _date, datetime as _dt, timedelta
        _td = lambda k: timedelta(days=k)
        from cfx.dates_bank import extract as _extract
        cd = _dt.fromisoformat(call["started_at"]).date()
        need = {f["valid_from"][:10]} | ({f["valid_to"][:10]} if f.get("valid_to") else set())
        # проверяем только восстановимые даты: не день звонка и не «скрытые» давние (старше 60 дней)
        need = {d for d in need if d != cd.isoformat() and (cd - _date.fromisoformat(d)).days <= 60}
        near = " ".join(turns[j]["text"] for j in range(max(0, ref.turn_idx - 1), min(len(turns), ref.turn_idx + 2)))
        have = {d for _, _, _, ds in _extract(near, cd) for d in ds}
        have |= {d + _td(k) for d in have for k in (-1, 1)}   # допуск ±1 день
        if need and not (need & {d.isoformat() for d in have}):   # хотя бы одна граница должна восстанавливаться из текста
            continue
        f["source"] = {"turn_idx": ref.turn_idx, "span": None, "quote": None}
        ok_facts.append(f)
    if len(ok_facts) < 0.8 * len(call["facts"]):
        return None
    return {**call, "turns": turns, "facts": ok_facts, "teacher": _env("TEACHER_MODEL", "deepseek-chat")}


def distill(calls, style, out_path, dry_run=None):
    """Переписать черновики учителем. dry_run=True — сохранить промпты без вызова API."""
    dry_run = (not _key()) if dry_run is None else dry_run
    n_ok = 0
    with open(out_path, "w") as fh:
        if dry_run:
            for c in calls:
                fh.write(json.dumps({"system": SYSTEM_DISTILL, "user": USER_DISTILL.format(
                    style=json.dumps(style, ensure_ascii=False), started_at=c["started_at"],
                    plan=json.dumps(plan_of(c), ensure_ascii=False, indent=0),
                    draft="\n".join(f"{t['idx']} {t['speaker']}: {t['text']}" for t in c["turns"]))}, ensure_ascii=False) + "\n")
            return {"mode": "dry_run", "prompts": len(calls)}
        chain = distill_chain()
        for c in calls:
            try:
                out = chain.invoke({"style": json.dumps(style, ensure_ascii=False), "started_at": c["started_at"],
                                    "plan": json.dumps(plan_of(c), ensure_ascii=False),
                                    "draft": "\n".join(f"{t['idx']} {t['speaker']}: {t['text']}" for t in c["turns"])})
                v = validate(c, out)
                if v:
                    fh.write(json.dumps(v, ensure_ascii=False) + "\n")
                    n_ok += 1
            except Exception as e:  # сеть/формат — пропускаем пример
                continue
    return {"mode": "api", "accepted": n_ok, "requested": len(calls)}


def extractor_runnable(extractor, genes=None):
    """Экстрактор как LangChain Runnable: invoke({'call_id','started_at','turns'}) → звонок с фактами."""
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(lambda call: extractor.predict(call, genes), name="CompactFactExtractor")


def verify_uncertain(call, facts, lo=0.3, hi=0.55, window=2):
    """Отправить в LLM только факты из «серой зоны» уверенности; вернуть отфильтрованные факты и число вызовов."""
    if not _key():
        return facts, 0
    chain = verifier_chain()
    kept, calls = [], 0
    for f in facts:
        c = f.get("confidence", 1.0)
        if lo <= c <= hi:
            ti = f["source"]["turn_idx"]
            snip = "\n".join(f"{t['speaker']}: {t['text']}" for t in call["turns"][max(0, ti - window): ti + window + 1])
            try:
                r = chain.invoke({"snippet": snip, "attribute": f["attribute"], "value": f["value"],
                                  "object": f["object"]["name"], "speaker": f["speaker"], "valid_from": f["valid_from"]})
                calls += 1
                if not r.supported:
                    continue
            except Exception:
                pass
        kept.append(f)
    return kept, calls
