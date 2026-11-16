"""Генератор v2: прицельные сценарии для дистилляции (блоки A–E) с осмысленными сочетаниями.

Каждый звонок = ПЛАН ФАКТОВ (эталон «по построению»: значения, объекты, даты, говорящий) + СКЕЛЕТ диалога.
Учитель-LLM (teacher2.py) превращает скелет в живой разговор с ASR-шумом заданного уровня и возвращает
для каждого факта реплику и точную цитату; код проверяет ответ.

Блоки (по разбору ошибок модели):
  A  несколько устройств: 2–3 объекта, у каждого своё состояние (и «до этого работало нормально»)
  B  даты начала: разнообразные относительные выражения, смена тарифа, цепочка статусов заявки
  C  перефразы: договорённости и действия должны быть сказаны своими словами, не как в каталоге
  D  ловушки schema.md §4: отказ, отвергнутая догадка, гипотеза, «у соседа», оператор зачитывает тарифы
  E  длинные реалистичные звонки: 20–30 реплик с уточнениями, ожиданием, повторами
Сочетания «объект ↔ состояние ↔ действие» берутся из совместной встречаемости в train (без «розетка + домофон»).
"""
from __future__ import annotations

import json, os, random, sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from generator import Generator  # noqa: E402

# только выражения, которые однозначно понимает банк дат (иначе эталонная дата будет неверной)
PAST_B = ["вчера", "позавчера", "вчера утром", "позавчера вечером", "со вторника", "с понедельника", "со среды",
          "с пятницы", "с субботы", "три дня назад", "два дня назад", "четыре дня назад", "дня три назад",
          "дней пять назад", "неделю назад", "на прошлой неделе"]
LONG_AGO = ["месяц назад", "в прошлом месяце"]
FUTURE_B = ["завтра", "послезавтра", "через три дня", "через два дня", "в четверг", "в пятницу", "в понедельник",
            "через неделю", "на следующей неделе"]
TIMES_B = ["", "", "после двух", "до обеда", "после обеда", "с утра", "к вечеру", "после трёх", "с десяти до двух"]
TRAPS = {
    "offer_rejected": "оператор предлагает услугу «{x}», клиент отказывается (это НЕ факт)",
    "wrong_guess": "оператор ошибочно предполагает тариф «{x}», клиент поправляет (тариф «{x}» — НЕ факт)",
    "hypothetical": "клиент рассуждает гипотетически «если бы я подключил {x}…» (НЕ факт)",
    "neighbor": "клиент упоминает, что у соседа/мамы {x} тоже не работает (про соседа — НЕ факт)",
    "op_reads": "оператор зачитывает список тарифов «{x}» и «{y}», клиент ничего не выбирает (НЕ факты)",
    "question": "клиент спрашивает «а может мне перейти на {x}?», но решения не принимает (НЕ факт)",
}
OFFERS = ["подписку на онлайн-кинотеатр", "пакет дополнительных гигабайтов", "антивирус для дома",
          "пакет детских каналов", "статический IP", "умную камеру"]
BLOCKS = [("A", 2000), ("B", 1500), ("C", 2000), ("D", 1500), ("E", 1000)]
NOISE = ["лёгкий", "средний", "сильный"]


class Generator2(Generator):
    def __init__(self, train_calls, catalog, markers, seed=7):
        super().__init__(train_calls, catalog, markers, seed)
        self.catalog = catalog
        st, ac = defaultdict(Counter), defaultdict(Counter)
        for c in train_calls:
            objs = {f["object"]["name"] for f in c["facts"] if f["object"]["type"] in ("service", "device") and f["object"]["name"]}
            for f in c["facts"]:
                if f["attribute"] == "service_state" and f["object"]["name"] in objs and f["value"] != "работает нормально":
                    st[f["object"]["name"]][f["value"]] += 1
                if f["attribute"] == "action":
                    for o in objs:
                        ac[o][f["value"]] += 1
        self.obj_states = {o: [v for v, n in cnt.most_common(25) if n >= 2] for o, cnt in st.items()}
        self.obj_actions = {o: [v for v, n in cnt.most_common(25) if n >= 2] for o, cnt in ac.items()}
        self.objs2 = [o for o in self.objs if len(self.obj_states.get(o, [])) >= 2]

    # ---------------------------------------------------------------- утилиты
    def _d(self, expr, d0):
        return self.date_of(expr, d0)

    def _iso(self, d):
        return datetime.combine(d, datetime.min.time()).isoformat() if not isinstance(d, str) else d

    def _fact(self, plan, obj, otype, attr, val, spk, beat, vf, vt=None, expr=None):
        plan.append({"object": {"id": f"{otype}:{obj}", "type": otype, "name": obj}, "attribute": attr, "value": val,
                     "polarity": -1 if val == "больше не используется" else 1, "speaker": spk,
                     "valid_from": self._iso(vf), "valid_to": self._iso(vt) if vt is not None else None,
                     "date_expr": expr, "beat": beat})

    # ---------------------------------------------------------------- звонок
    def call(self, idx, block):
        r = self.rng
        cd = datetime(2026, r.randint(1, 10), r.randint(1, 28), r.choice([9, 10, 11, 14, 16, 18]))
        d0 = cd.date()
        beats, plan, traps, notes = [], [], [], []
        say = lambda spk, text: beats.append({"speaker": spk, "text": text}) or len(beats) - 1
        say("operator", "Приветствие, оператор представляется.")

        def device_story(obj, with_ok=None, expr_pool=PAST_B):
            val = r.choice(self.obj_states.get(obj) or self.vals["service_state"])
            ex = r.choice(expr_pool)
            ds = self._d(ex, d0)
            b = say("client", f"Клиент сообщает, что {ex} перестал пользоваться услугой «{obj}» (больше не использует)."
                    if val == "больше не используется" else f"Клиент описывает проблему: у него {obj} — «{val}», началось {ex}.")
            if with_ok if with_ok is not None else r.random() < 0.4:
                b2 = say("client", f"Клиент уточняет, что до этого {obj} работал нормально.")
                self._fact(plan, obj, "service", "service_state", "работает нормально", "client", b2,
                           d0 - timedelta(days=r.randint(60, 300)), ds, None)
            self._fact(plan, obj, "service", "service_state", val, "client", b, ds, None, ex)
            return val

        def actions(obj, k, paraphrase=False):
            pool = self.obj_actions.get(obj) or self.vals["action"]
            for act in r.sample(pool, min(k, len(pool))):
                b = say("client", f"Клиент рассказывает, что уже сделал: «{act}»" + (" (сказать своими словами!)" if paraphrase else "") + ".")
                self._fact(plan, "клиент", "client", "action", act, "client", b, d0)

        def agreement(paraphrase=False):
            agr = r.choice(self.vals["agreement"])
            from cfx.dates_bank import extract as _ex
            own = _ex(agr, d0)                       # дата уже внутри значения («мастер приедет в четверг»)
            if own:
                t_, a_, b_, ds_ = own[0]
                ex, tm = agr[a_:b_], ""
                say("operator", f"Оператор предлагает: «{agr}»" + (" (сказать своими словами, дату «{ex}» сохранить!)" if paraphrase else "") + ".")
            else:
                ex, tm = r.choice(FUTURE_B), r.choice(TIMES_B)
                say("operator", f"Оператор предлагает: «{agr}», {ex} {tm}".strip() + (" (сказать своими словами!)" if paraphrase else "") + ".")
            dag = self._d(ex, d0)
            b = say("client", "Клиент соглашается.")
            self._fact(plan, "клиент", "client", "agreement", agr, "client", b, dag, None, ex)

        def tariff_switch():
            new, old = r.sample(self.vals["tariff"], 2)
            ex = r.choice(LONG_AGO + ["на прошлой неделе", "неделю назад"])
            dsw = self._d(ex, d0)
            b = say("client", f"Клиент говорит, что {ex} перешёл с тарифа «{old}» на «{new}».")
            self._fact(plan, "клиент", "client", "tariff", old, "client", b, d0 - timedelta(days=r.randint(120, 400)), dsw, None)
            self._fact(plan, "клиент", "client", "tariff", new, "client", b, dsw, None, ex)

        def ticket():
            num = str(r.randint(4000, 6000))
            st = r.choice(["в работе", "назначен выезд специалиста", "выполнена"])
            e1, e2 = r.choice([("с понедельника", "вчера"), ("со вторника", "вчера"), ("три дня назад", "позавчера"),
                               ("неделю назад", "два дня назад"), ("на прошлой неделе", "вчера")])
            d1, d2 = self._d(e1, d0), self._d(e2, d0)
            b = say("operator", f"Оператор сообщает по заявке {num}: принята {e1}, а с {e2} статус «{st}».")
            self._fact(plan, f"заявка {num}", "ticket", "ticket_status", "принята", "operator", b, d1, d2, e1)
            self._fact(plan, f"заявка {num}", "ticket", "ticket_status", st, "operator", b, d2, None, e2)

        objs = self.objs2 or self.objs
        if block == "A":
            for o in r.sample(objs, r.choice([2, 2, 3])):
                device_story(o)
            say("operator", "Оператор уточняет, какое устройство беспокоит больше, задаёт вопросы по каждому.")
            actions(r.choice(objs), r.choice([0, 1, 2]))
            if r.random() < 0.7: agreement()
        elif block == "B":
            device_story(r.choice(objs), with_ok=r.random() < 0.6)
            if r.random() < 0.7: tariff_switch()
            if r.random() < 0.6: ticket()
            agreement()
        elif block == "C":
            o = r.choice(objs)
            device_story(o)
            actions(o, r.choice([2, 3]), paraphrase=True)
            agreement(paraphrase=True)
            if r.random() < 0.4: agreement(paraphrase=True)
        elif block == "D":
            o = r.choice(objs)
            device_story(o)
            for t in r.sample(list(TRAPS), r.choice([2, 3])):
                x, y = r.sample(self.vals["tariff"], 2) if t in ("wrong_guess", "op_reads", "question") else (
                    r.choice(OFFERS) if t in ("offer_rejected", "hypothetical") else o, None)
                b = say("operator" if t in ("offer_rejected", "wrong_guess", "op_reads") else "client",
                        "ЛОВУШКА: " + TRAPS[t].format(x=x, y=y))
                traps.append(t)
            if r.random() < 0.6: agreement()
        else:  # E
            o = r.choice(objs)
            device_story(o)
            actions(o, r.choice([1, 2]))
            if r.random() < 0.4: tariff_switch()
            if r.random() < 0.5: ticket()
            agreement()
            notes.append("растяни звонок до 20–30 реплик: проверка личности, «минутку, проверяю», повторные вопросы, "
                         "отвлечения клиента, уточнение адреса без реальных данных")
        say("operator", "Оператор прощается.")
        return {"call_id": f"distill_{block}_{idx:05d}", "started_at": cd.isoformat(), "duration_sec": None,
                "series": None, "block": block, "noise": r.choice(NOISE), "beats": beats, "plan": plan,
                "traps": traps, "notes": notes}


def build(n_scale=1.0, seed=7):
    here = os.path.join(os.path.dirname(__file__), "..", "..")
    train = [json.loads(l) for l in open(os.path.join(here, "data/train.jsonl"), encoding="utf-8")]
    g = Generator2(train, json.load(open(os.path.join(here, "data/catalog.json"))),
                   json.load(open(os.path.join(here, "runs/markers.json"))), seed)
    out = []
    for b, n in BLOCKS:
        out += [g.call(i, b) for i in range(int(n * n_scale))]
    random.Random(seed).shuffle(out)
    return out


if __name__ == "__main__":
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    path = sys.argv[2] if len(sys.argv) > 2 else "runs/plans_v2.jsonl"
    calls = build(scale)
    with open(path, "w", encoding="utf-8") as fh:
        for c in calls:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("written", len(calls), path, Counter(c["block"] for c in calls))
