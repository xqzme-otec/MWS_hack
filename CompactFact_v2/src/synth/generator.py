"""Вероятностный генератор синтетических звонков с разметкой «по построению».

1) План фактов сэмплируется из эмпирических распределений train (значения, объекты, наличие
   переходов тарифа, цепочки статусов, ловушки), даты — из банка относительных выражений,
   абсолютная дата считается тем же правилом, что у оценщика.
2) Реализация: шаблоны реплик клиента/оператора со слотами; маркеры разговорной речи
   вставляются с частотами, измеренными на реальных телефонных транскриптах (open_stt) и корпусе.
3) ASR-шум: замены букв (таблица из пар «чистая цитата ↔ зашумлённая реплика»), выпадения слов,
   обрывы реплик; уровень шума — параметр (0.25 как в train, 0.4+ — «adversarial»).
Этот же план фактов + черновик диалога — вход учителя-LLM (teacher.py) для перефразирования.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cfx.dates_bank import extract

SUBS = [("о", "а", 986), ("е", "и", 766), ("а", "о", 260), ("д", "т", 237), ("о", "е", 228), ("а", "е", 220),
        ("е", "а", 201), ("в", "ф", 168), ("б", "п", 159), ("о", "и", 151), ("а", "у", 134), ("а", "и", 133),
        ("з", "с", 80), ("г", "к", 70), ("ж", "ш", 60), ("ч", "щ", 40), ("т", "д", 60), ("с", "з", 50)]

PAST = ["вчера", "позавчера", "со вторника", "с понедельника", "со среды", "с пятницы", "три дня назад", "два дня назад",
        "неделю назад", "на прошлой неделе", "месяц назад", "в прошлом месяце", "пять дней назад"]
FUTURE = ["завтра", "послезавтра", "через три дня", "через два дня", "в четверг", "в пятницу", "в понедельник", "через неделю"]
TIMES = ["", "", " после двух", " до обеда", " после обеда", " с утра", " к вечеру", " после трёх"]

T_CLIENT_PROBLEM = ["У меня {obj} {val}, {date} это началось.", "{date} у меня {obj} — {val}, вообще ничего не понимаю.",
                    "Слушайте, {obj} {val}. Началось {date}.", "Звоню по поводу {obj}: {val}, {date} ещё."]
T_CLIENT_OK_BEFORE = ["А до этого {obj} работал нормально.", "Раньше всё работало нормально, {obj} вообще без проблем.",
                      "Ещё {date2} всё работало нормально."]
T_CLIENT_ACTION = ["Я уже {act}, не помогло.", "{act} делала, толку ноль.", "Да я {act} уже, ничего не изменилось.",
                   "Пробовала — {act}. Без результата."]
T_TARIFF_SWITCH = ["Да нет, я {date} на «{new}» перешла, а до этого был «{old}».", "У меня сейчас «{new}», {date} поменяла. «{old}» был раньше.",
                   "Был «{old}», а {date} сменила на «{new}»."]
T_TARIFF_SIMPLE = ["У меня тариф «{new}».", "Тариф у меня «{new}», да.", "Я на «{new}» сижу давно."]
T_OP_GUESS = ["У вас тариф «{wrong}», верно?", "Вижу, у вас «{wrong}» подключён?"]
T_TICKET = ["Заявка {num} у вас принята {d1}.", "По заявке {num}: принята {d1}, а сейчас статус «{st}».",
            "Вижу заявку {num} — до {d2} она была «принята», теперь «{st}»."]
T_OFFER = ["Могу предложить вам {offer}, хотите?", "Можем подключить {offer}, интересно?"]
T_REJECT = ["Нет, спасибо, не надо.", "Да нет, мне это не нужно.", "Не, не надо ничего подключать."]
T_AGREE_OP = ["Давайте так: {agr} {date}{time}. Вам удобно?", "Могу оформить: {agr}, {date}{time}. Подходит?"]
T_AGREE_CL = ["Да, подходит.", "Хорошо, давайте.", "Да, договорились.", "Ну давайте, удобно."]
OFFERS = ["подписку на онлайн-кинотеатр", "пакет дополнительных гигабайтов", "антивирус для дома", "пакет детских каналов"]


class Generator:
    def __init__(self, train_calls, catalog, markers, seed=42):
        self.rng = random.Random(seed)
        c = Counter
        self.vals = {a: c() for a in ("service_state", "tariff", "action", "agreement")}
        self.objs = c()
        for call in train_calls:
            for f in call["facts"]:
                if f["attribute"] in self.vals and f["value"] != "работает нормально":
                    self.vals[f["attribute"]][f["value"]] += 1
                if f["object"]["type"] in ("service", "device") and f["object"]["name"]:
                    self.objs[f["object"]["name"]] += 1
        self.vals["tariff"].update({t: 3 for t in catalog["tariffs"]})
        self.vals["agreement"].update({t: 10 for t in catalog["agreements"]})
        for a in self.vals:
            self.vals[a] = [v for v, n in self.vals[a].most_common() if n >= 2 and len(v.split()) <= 6]
        self.objs = [o for o, n in self.objs.most_common() if n >= 3]
        self.mk = markers

    # -------------------- шум и маркеры --------------------
    def markers(self, text, speaker):
        st = self.mk["case_client" if speaker == "client" else "case_operator"]
        r = self.rng
        for m, p in sorted(st["p_first"].items(), key=lambda x: -x[1])[:8]:
            if r.random() < p * 0.8:
                text = m.capitalize() + ", " + text[0].lower() + text[1:]
                break
        if speaker == "client" and r.random() < 0.15:
            ws = text.split()
            k = r.randrange(1, max(2, len(ws)))
            ws.insert(k, r.choice(["эээ,", "ну,", "вот,", "короче,", "как бы,"]))
            text = " ".join(ws)
        return text

    def noise(self, text, level):
        r = self.rng
        tot = sum(n for _, _, n in SUBS)

        def corrupt(w):
            if len(w) < 4 or r.random() > level:
                return w
            chars = list(w)
            for _ in range(1 + (r.random() < level)):
                x = r.random() * tot
                for a, b, n in SUBS:
                    x -= n
                    if x <= 0:
                        idx = [i for i, ch in enumerate(chars) if ch == a]
                        if idx:
                            chars[r.choice(idx)] = b
                        break
            return "".join(chars)
        ws = [corrupt(w) for w in text.split() if r.random() > level * 0.08]
        out = " ".join(ws)
        if r.random() < level * 0.15:
            out = " ".join(out.split()[: max(3, int(len(out.split()) * 0.7))]) + " —"
        return out

    # -------------------- даты --------------------
    def date_of(self, expr, cd):
        hits = extract(expr, cd)
        if not hits:
            return cd
        t, a, b, ds = hits[0]
        if t == "on_wd":
            return ds[1] if any(w in expr for w in ("в четверг", "в пятницу", "в понедельник")) and expr in FUTURE else ds[0]
        return self.rng.choice(ds)

    # -------------------- звонок --------------------
    def call(self, idx, noise_level=0.25):
        r = self.rng
        cd = datetime(2026, r.randint(1, 10), r.randint(1, 28), r.choice([9, 10, 11, 14, 16, 18]))
        d0 = cd.date()
        turns, facts = [], []

        def say(spk, text):
            turns.append({"idx": len(turns), "speaker": spk, "text": text})
            return len(turns) - 1

        def fact(obj, attr, val, spk, ti, vf, vt=None, pol=1, otype="service"):
            facts.append({"object": {"id": f"syn{idx}:{otype}:{obj}", "type": otype, "name": obj}, "attribute": attr,
                          "value": val, "polarity": pol, "speaker": spk,
                          "valid_from": vf if isinstance(vf, str) else datetime.combine(vf, datetime.min.time()).isoformat(),
                          "valid_to": (vt if isinstance(vt, str) or vt is None else datetime.combine(vt, datetime.min.time()).isoformat()),
                          "source": {"turn_idx": ti, "span": None, "quote": None}})
        name = r.choice(["Марина Олеговна", "Елена Петровна", "Оля Викторовна", "Катерина Сергеевна"])
        say("operator", "Здравствуйте! Техническая поддержка, меня зовут " + r.choice(["Марина", "Анна", "Ольга"]) + ", чем могу помочь?")
        obj = r.choice(self.objs)
        val = r.choice(self.vals["service_state"])
        dexpr = r.choice(PAST)
        dstart = self.date_of(dexpr, d0)
        ti = say("client", r.choice(T_CLIENT_PROBLEM).format(obj=obj, val=val, date=dexpr))
        if r.random() < 0.5:
            tj = say("client", r.choice(T_CLIENT_OK_BEFORE).format(obj=obj, date2=dexpr))
            fact(obj, "service_state", "работает нормально", "client", tj, d0 - timedelta(days=r.randint(60, 300)), dstart)
        fact(obj, "service_state", val, "client", ti, dstart)
        say("operator", f"{name}, понимаю. Давайте разберёмся. Что вы уже пробовали сделать?")
        for _ in range(r.choice([0, 1, 1, 2])):
            act = r.choice(self.vals["action"])
            ti = say("client", r.choice(T_CLIENT_ACTION).format(act=act.replace(" — не помогло", "")))
            fact("клиент", "action", act, "client", ti, d0, otype="client")
        if r.random() < 0.35:
            new, old = r.sample(self.vals["tariff"], 2)
            if r.random() < 0.3:  # ловушка: неверная догадка оператора
                say("operator", r.choice(T_OP_GUESS).format(wrong=old))
            if r.random() < 0.6:
                de = r.choice(["месяц назад", "в прошлом месяце", "на прошлой неделе", "неделю назад"])
                dsw = self.date_of(de, d0)
                ti = say("client", r.choice(T_TARIFF_SWITCH).format(new=new, old=old, date=de))
                fact("клиент", "tariff", old, "client", ti, d0 - timedelta(days=r.randint(120, 400)), dsw, otype="client")
                fact("клиент", "tariff", new, "client", ti, dsw, otype="client")
            else:
                ti = say("client", r.choice(T_TARIFF_SIMPLE).format(new=new))
                fact("клиент", "tariff", new, "client", ti, cd.isoformat(), otype="client")
        if r.random() < 0.45:
            num = str(r.randint(4000, 6000))
            st = r.choice(["в работе", "назначен выезд специалиста", "выполнена"])
            d1e, d2e = r.choice([("с понедельника", "вчера"), ("со вторника", "вчера"), ("три дня назад", "позавчера")])
            dd1, dd2 = self.date_of(d1e, d0), self.date_of(d2e, d0)
            ti = say("operator", r.choice(T_TICKET).format(num=num, d1=d1e, d2=d2e, st=st))
            fact(f"заявка {num}", "ticket_status", "принята", "operator", ti, dd1, dd2, otype="ticket")
            fact(f"заявка {num}", "ticket_status", st, "operator", ti, dd2, otype="ticket")
        if r.random() < 0.4:  # ловушка: неподтверждённое предложение
            say("operator", r.choice(T_OFFER).format(offer=r.choice(OFFERS)))
            say("client", r.choice(T_REJECT))
        if r.random() < 0.85:
            agr = r.choice(self.vals["agreement"])
            de, tm = r.choice(FUTURE), r.choice(TIMES)
            dag = self.date_of(de, d0)
            say("operator", r.choice(T_AGREE_OP).format(agr=agr, date=de, time=tm))
            ti = say("client", r.choice(T_AGREE_CL))
            fact("клиент", "agreement", agr, "client", ti, dag, otype="client")
        say("operator", "Спасибо за обращение, всего доброго!")
        for t in turns:
            t["text"] = self.noise(self.markers(t["text"], t["speaker"]), noise_level)
        return {"call_id": f"synthetic_{idx:06d}:call0000", "started_at": cd.isoformat(), "duration_sec": 60 * len(turns),
                "series": None, "turns": turns, "noise_level": noise_level, "facts": facts, "synthetic": True}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    n, out = int(sys.argv[1]), sys.argv[2]
    level = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    train = [json.loads(l) for l in open("data/train.jsonl")]
    g = Generator(train, json.load(open("data/catalog.json")), json.load(open("runs/markers.json")))
    with open(out, "w") as fh:
        for i in range(n):
            fh.write(json.dumps(g.call(i, level), ensure_ascii=False) + "\n")
    print("written", n, out)
