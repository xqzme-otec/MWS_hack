"use strict";
const ATTR = {
  service_state: "состояние услуги", tariff: "тариф", ticket_status: "статус заявки",
  action: "действие", agreement: "договорённость",
};
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const day = (s) => (s ? s.slice(0, 10) : "");
const fmtDay = (s) => { if (!s) return ""; const [y, m, d] = s.slice(0, 10).split("-"); return `${d}.${m}.${y}`; };

let state = { mode: "hybrid", items: [], cur: null, call: null, res: null };

// ---------- шапка: метрики из reports/summary.json
async function loadReport() {
  try {
    const r = await (await fetch("/api/report")).json();
    const h = r.headline || {};
    const chips = [];
    if (h.temporal_recall != null) chips.push(`<span class="chip good">temporal_recall <b>${h.temporal_recall.toFixed(3)}</b></span>`);
    if (h.state_accuracy != null) chips.push(`<span class="chip good">state_accuracy <b>${h.state_accuracy.toFixed(3)}</b></span>`);
    if (h.params_m) chips.push(`<span class="chip">параметры <b>${h.params_m} M</b></span>`);
    if (h.latency_ms_p50) chips.push(`<span class="chip">CPU p50 <b>${Math.round(h.latency_ms_p50)} мс</b></span>`);
    chips.push(`<span class="chip">GPU <b>0 с</b></span>`);
    $("#chips").innerHTML = chips.join("");
  } catch (e) { /* без сводки — просто без чипов */ }
}

// ---------- список звонков
async function loadList() {
  const r = await (await fetch("/api/examples")).json();
  state.items = r.items;
  $("#src").textContent = r.source ? `источник: ${r.source} · ${r.items.length} звонков` : "нет примеров — используйте «Свой звонок»";
  renderList();
}
function renderList() {
  const q = $("#q").value.trim().toLowerCase();
  $("#list").innerHTML = state.items
    .filter((it) => !q || it.call_id.toLowerCase().includes(q))
    .slice(0, 400)
    .map((it) => `<li data-i="${it.i}" class="${state.cur === it.i ? "on" : ""}">
        <span class="id">${esc(it.call_id)}</span>
        <span class="meta">${it.n_turns} репл. · ${it.n_facts} факт.</span></li>`).join("");
}
$("#list").addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) openExample(+li.dataset.i); });
$("#q").addEventListener("input", renderList);
$("#rand").addEventListener("click", () => { if (state.items.length) openExample(state.items[Math.floor(Math.random() * state.items.length)].i); });

async function openExample(i) {
  state.cur = i; renderList();
  const call = await (await fetch(`/api/examples/${i}`)).json();
  run(call);
}

// ---------- свой звонок
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === b));
  $("#pane-list").classList.toggle("hidden", b.dataset.tab !== "list");
  $("#pane-own").classList.toggle("hidden", b.dataset.tab !== "own");
}));
$("#own-run").addEventListener("click", () => {
  const turns = $("#own-text").value.split("\n").map((l) => l.trim()).filter(Boolean).map((l, i) => {
    const m = l.match(/^(ОП|КЛ|оператор|клиент|op|cl)\s*[:：-]\s*(.*)$/i);
    const op = m ? /^(оп|оператор|op)$/i.test(m[1]) : i % 2 === 0;
    return { idx: i, speaker: op ? "operator" : "client", text: m ? m[2] : l };
  });
  const d = $("#own-date").value || "2026-01-21T10:00";
  state.cur = null; renderList();
  run({ call_id: "my_call", started_at: d.length === 16 ? d + ":00" : d, turns });
});

// ---------- движок
document.querySelectorAll(".engine button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".engine button").forEach((x) => x.classList.toggle("on", x === b));
  state.mode = b.dataset.mode;
  if (state.call) run(state.call);
}));
$("#dbg").addEventListener("change", () => state.res && render());

// ---------- запуск
async function run(call) {
  state.call = call;
  $("#latency").innerHTML = `<span class="spin"></span> считаю…`;
  const hasGold = Array.isArray(call.facts) && call.facts.length > 0;
  const url = hasGold ? `/api/score?mode=${state.mode}` : `/extract?mode=${state.mode}&debug=1`;
  const resp = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(call) });
  if (!resp.ok) { $("#latency").textContent = "ошибка " + resp.status; return; }
  const j = await resp.json();
  state.res = hasGold ? j : { prediction: j, metrics: null, matches: [] };
  render();
}

function render() {
  const call = state.call, res = state.res, pred = res.prediction, facts = pred.facts;
  const matchedPred = new Map(res.matches.map((m) => [m.pred, m.gold]));
  const matchedGold = new Set(res.matches.map((m) => m.gold));
  const hasGold = Array.isArray(call.facts) && call.facts.length > 0;

  $("#call-title").textContent = call.call_id + (call.synthetic ? " · синтетический" : "");
  $("#call-sub").textContent = `${fmtDay(call.started_at)} · ${call.turns.length} реплик` + (hasGold ? ` · в эталоне ${call.facts.length} фактов` : "");
  $("#legend").innerHTML = Object.entries(ATTR).map(([k, v]) => `<span class="lg c-${k}">${v}</span>`).join("");

  // ----- транскрипт с подсветкой
  const marks = {};
  facts.forEach((f, i) => {
    const s = f.source || {};
    if (s.span) (marks[s.turn_idx] ||= []).push({ a: s.span[0], b: s.span[1], cls: `a-${f.attribute}`, i, tip: `${ATTR[f.attribute]}: ${f.value}` });
  });
  if ($("#dbg").checked) (pred.meta.gliner_spans || []).forEach((g) => {
    if (g.span && ATTR[g.label]) (marks[g.turn_idx] ||= []).push({ a: g.span[0], b: g.span[1], cls: "raw", i: -1, tip: `GLiNER ${g.label} ${g.score}` });
  });
  $("#transcript").innerHTML = pred.turns.map((t) => {
    const ms = (marks[t.idx] || []).sort((x, y) => x.a - y.a || (y.i >= 0) - (x.i >= 0));
    let html = "", pos = 0;
    for (const m of ms) {
      if (m.a < pos || m.b <= m.a) continue;
      html += esc(t.text.slice(pos, m.a)) + `<mark class="${m.cls}" data-f="${m.i}" title="${esc(m.tip)}">${esc(t.text.slice(m.a, m.b))}</mark>`;
      pos = m.b;
    }
    html += esc(t.text.slice(pos));
    const cl = t.speaker === "client";
    return `<div class="turn ${cl ? "client" : ""}" id="t${t.idx}"><div class="who">${cl ? "КЛ" : "ОП"}</div>
      <div class="bubble"><div class="idx">#${t.idx}</div>${html}</div></div>`;
  }).join("");

  // ----- задержка
  const tm = pred.meta.timing_ms || {}, tot = tm.total_ms || pred.meta.latency_ms;
  const seg = (v, c) => (v ? `<span style="width:${(100 * v) / tot}%;background:${c}"></span>` : "");
  $("#latency").innerHTML = `<b>${Math.round(tot)} мс</b> · ${pred.meta.engine}<br>
    GLiNER: ${pred.meta.turns_gliner}/${pred.meta.turns_total} реплик
    <div class="bar">${seg(tm.gliner_ms, "#e30611")}${seg(tm.boosting_ms, "#6b7280")}</div>`;
  $("#nfacts").textContent = facts.length;

  // ----- метрики звонка
  const cm = $("#callmetrics");
  if (res.metrics) {
    const m = res.metrics, f2 = (x) => (x == null ? "—" : x.toFixed(2));
    cm.innerHTML = [["полнота", m.fact_recall], ["точность", m.fact_precision], ["даты", m.temporal_recall], ["текущее сост.", m.state_accuracy]]
      .map(([k, v]) => `<div class="m"><div class="k">${k}</div><div class="v">${f2(v)}</div></div>`).join("");
    cm.classList.remove("hidden");
  } else cm.classList.add("hidden");

  // ----- таймлайн
  renderTimeline(facts, call.started_at);

  // ----- карточки фактов
  $("#facts").innerHTML = facts.length ? facts.map((f, i) => {
    const st = hasGold ? (matchedPred.has(i) ? `<span class="status ok">✓ в эталоне</span>` : `<span class="status no">лишний</span>`) : "";
    return card(f, i, st);
  }).join("") : `<div class="empty">фактов не найдено</div>`;

  // ----- пропущенные
  const missed = hasGold ? call.facts.map((f, i) => [f, i]).filter(([, i]) => !matchedGold.has(i)) : [];
  const mi = $("#missed");
  if (missed.length) {
    mi.innerHTML = `<h4>Пропущено из эталона: ${missed.length}</h4>` + missed.map(([f]) => card(f, -1, `<span class="status no">пропуск</span>`)).join("");
    mi.classList.remove("hidden");
  } else mi.classList.add("hidden");

  $("#json").textContent = JSON.stringify({ call_id: pred.call_id, facts: pred.facts }, null, 1);
}

function card(f, i, status) {
  const s = f.source || {}, to = f.valid_to ? ` → ${fmtDay(f.valid_to)}` : (["service_state", "tariff", "ticket_status"].includes(f.attribute) ? " → сейчас" : "");
  return `<div class="fact c-${f.attribute}" data-turn="${s.turn_idx}" data-f="${i}">
    <div class="top1"><span class="attr">${ATTR[f.attribute] || f.attribute}</span>${status}</div>
    <div class="val">${f.polarity === -1 ? '<span class="neg">НЕ </span>' : ""}${esc(f.value)}</div>
    <div class="sub"><span>👤 ${esc(f.object?.name || "")}</span><span>📅 ${fmtDay(f.valid_from)}${to}</span><span>💬 реплика #${s.turn_idx}</span>${f.speaker ? `<span>${f.speaker === "client" ? "клиент" : "оператор"}</span>` : ""}</div>
  </div>`;
}

document.addEventListener("click", (e) => {
  const el = e.target.closest(".fact, mark");
  if (!el) return;
  const ti = el.dataset.turn ?? state.res?.prediction.facts[+el.dataset.f]?.source?.turn_idx;
  const t = document.getElementById("t" + ti);
  if (!t) return;
  document.querySelectorAll(".turn.hl").forEach((x) => x.classList.remove("hl"));
  t.classList.add("hl"); t.scrollIntoView({ behavior: "smooth", block: "center" });
});

function renderTimeline(facts, started) {
  const tl = $("#timeline");
  const fs = facts.filter((f) => f.attribute !== "action" && f.valid_from);
  if (!fs.length) { tl.classList.add("hidden"); return; }
  const ds = fs.flatMap((f) => [day(f.valid_from), day(f.valid_to)]).filter(Boolean).concat([day(started)]);
  const t = (s) => new Date(s + "T00:00:00").getTime();
  let lo = Math.min(...ds.map(t)), hi = Math.max(...ds.map(t));
  if (hi - lo < 4 * 864e5) { lo -= 2 * 864e5; hi += 2 * 864e5; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const x = (s) => (100 * (t(day(s)) - lo)) / (hi - lo);
  const now = x(started);
  tl.innerHTML = fs.map((f) => {
    const a = x(f.valid_from);
    let seg;
    if (f.attribute === "agreement") seg = `<span class="tl-dot" style="left:${a}%"></span>`;
    else if (f.valid_to) seg = `<span class="tl-seg" style="left:${a}%;width:${Math.max(1.5, x(f.valid_to) - a)}%"></span>`;
    else seg = `<span class="tl-seg open" style="left:${a}%;width:${Math.max(4, 100 - a)}%"></span>`;
    return `<div class="tl-row c-${f.attribute}"><div class="tl-lbl" title="${esc(f.value)}">${esc(f.value)}</div>
      <div class="tl-track"><span class="tl-now" style="left:${now}%"></span>${seg}</div></div>`;
  }).join("") + `<div class="tl-axis"><div></div><div><span>${fmtDay(new Date(lo).toISOString())}</span><span>звонок ${fmtDay(started)}</span><span>${fmtDay(new Date(hi).toISOString())}</span></div></div>`;
  tl.classList.remove("hidden");
}

loadReport();
loadList().then(() => { if (state.items.length) openExample(state.items[0].i); });
