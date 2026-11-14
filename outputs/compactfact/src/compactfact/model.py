fit_repr = normalize(.75 * fit_x + .25 * fit_context)
rank_x, rank_context = self._vectors(rank_rows)
dev_x, dev_context = self._vectors(dev_rows)
diagnostics = {}
for attr, bank in self.banks.items():
    print(f"[{self.config_name}] training {attr}: {len(bank['values'])} candidates", flush=True)
    bank["lexical"] = self.vectorizer.transform(bank["values"])
    rids, cids = [], []
    for i, labels in enumerate(fit_labels):
        for f in labels:
            if f["attribute"] == attr and norm(f["value"]) in bank["index"]:
                j = bank["index"][norm(f["value"])]
                rids.append(j);
                cids.append(i);
                bank["counts"][j] += 1
    assignment = sparse.csr_matrix((np.ones(len(rids), dtype=np.float32), (rids, cids)),
                                   shape=(len(bank["values"]), len(fit_rows)))
    proto = normalize(assignment @ fit_repr)
    bank["prototypes"] = normalize(.8 * proto + .2 * bank["lexical"]).astype(np.float32)
    features, _, _, y = self._candidates(attr, rank_rows, rank_x, rank_context, rank_labels)
    clf = HistGradientBoostingClassifier(max_iter=self.config["iterations"], max_leaf_nodes=15, max_depth=5,
                                         l2_regularization=2, learning_rate=.08, min_samples_leaf=25,
                                         early_stopping=False, random_state=self.seed)
    clf.fit(features, y)
    bank["ranker"] = clf
    df, dr, dc, _ = self._candidates(attr, dev_rows, dev_x, dev_context)
    probs = clf.predict_proba(df)[:, 1]
    truth = {(i, bank["index"][norm(f["value"])]) for i, labs in enumerate(dev_labels) for f in labs
             if f["attribute"] == attr and norm(f["value"]) in bank["index"]}
    n_gold = sum(f["attribute"] == attr for labs in dev_labels for f in labs)
    results = []
    for threshold in np.arange(.10, .81, .05):
        selected = {(int(i), int(j)) for i, j, p in zip(dr, dc, probs) if p >= threshold}
        tp = len(selected & truth)
        f1 = 2 * tp / max(len(selected) + n_gold, 1)
        results.append((f1, float(threshold), tp, len(selected)))
    best = max(results, key=lambda r: (r[0], r[1]))
    self.thresholds[attr] = round(best[1], 3)
    diagnostics[attr] = {"threshold": self.thresholds[attr], "canonical_source_dev_f1": round(best[0], 4),
                         "n_candidates": len(bank["values"]), "fit_supported_facts": int(bank["counts"].sum()),
                         "rank_positive_pairs": int(y.sum()), "dev_total_facts": n_gold,
                         "dev_supported_facts": len(truth)}
self.metadata = {"architecture": "char_tfidf_supervised_prototypes_gradient_boosted_ranker", "config": self.config_name,
                 "seed": self.seed, "n_features": len(self.vectorizer.vocabulary_),
                 "splits": {k: len(v) for k, v in groups.items()},
                 "training_seconds": round(time.perf_counter() - started, 3), "diagnostics": diagnostics,
                 "validation_used_for_training": False, "unknown_freeform_values": "not generated",
                 "date_fallback": "call timestamp is observation time; unknown historical start stays null"}
return self


@threadpool_limits.wrap(limits=2)
def raw_predictions(self, call):
    rows = turns_as_rows(inference_call(call))
    if not rows:
        return []
    x, context = self._vectors(rows)
    predictions = []
    for attr, bank in self.banks.items():
        features, rowids, candidateids, _ = self._candidates(attr, rows, x, context)
        probs = bank["ranker"].predict_proba(features)[:, 1]
        for i, j, prob in zip(rowids, candidateids, probs):
            if prob >= self.thresholds[attr]:
                predictions.append((attr, bank["values"][j], float(prob), rows[i]))
    return predictions


def _date(self, call, attr, value, row):
    text = row["text"]
    n = norm(text)
    historical = bool(re.search(r"раньше|до этого|был[ао]? |работал[аои]? |прежн|старый тариф", n))
    if value == "работает нормально" and historical:
        # No fabricated 180-day prior date to imitate hidden generator state.
        return None
    if attr == "tariff" and historical and re.search(r"переш|смен|вместо|поменял", n):
        v = norm(value)
        if re.search(r"(?:с тарифа|тариф был|раньше был|был тариф)\s*" + re.escape(v), n):
            return None
    date = normalize_date(text, call["started_at"], future=attr == "agreement")
    if date:
        return date
    # A short confirmation can inherit an immediately proposed deadline.
    if attr == "agreement" and len(n) < 130 and row["speaker"] == "client":
        date = normalize_date(row["prev"], call["started_at"], future=True)
        if date:
            return date
    if historical and attr in {"tariff", "ticket_status"}:
        return None
    return call["started_at"]


def predict(self, call):
    clean = inference_call(call)
    candidates = []
    for attr, value, probability, row in self.raw_predictions(clean):
        obj = self.resolver.resolve(clean, row["turn_idx"], attr, value)
        polarity = -1 if value == "больше не используется" else 1
        text = row["text"]
        # A literal full source turn is a valid quote. No artificial broad
        # span is supplied to inflate the official overlap-only metric.
        fact = {"object": obj, "attribute": attr, "value": value, "polarity": polarity,
                "speaker": row["speaker"], "valid_from": self._date(clean, attr, value, row), "valid_to": None,
                "source": {"turn_idx": row["turn_idx"], "span": None, "quote": text}}
        candidates.append((probability, fact))
    # Repetitions are merged, but the same state recurring at distinct known
    # times is retained. The most confident observed source wins.
    unique = {}
    for probability, fact in candidates:
        key = (fact["object"]["id"], fact["attribute"], norm(fact["value"]), fact["polarity"],
               (fact["valid_from"] or "")[:10])
        if key not in unique or probability > unique[key][0]:
            unique[key] = (probability, fact)
    facts = [f for _, f in unique.values()]
    # Unknown historical starts may be closed by a later observed transition.
    for fact in facts:
        if fact["valid_from"] is None and fact["attribute"] in {"service_state", "tariff", "ticket_status"}:
            following = [f for f in facts if f["object"]["id"] == fact["object"]["id"]
                         and f["attribute"] == fact["attribute"] and f["value"] != fact["value"]
                         and f["polarity"] == 1 and f["valid_from"] is not None
                         and f["source"]["turn_idx"] >= fact["source"]["turn_idx"]]
            if following:
                fact["valid_to"] = min(f["valid_from"] for f in following)
    facts = close_intervals(facts)
    clean["facts"] = sorted(facts, key=lambda f: (f["source"]["turn_idx"], f["attribute"], f["value"]))
    return clean


def save(self, directory):
    directory = Path(directory);
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(self, directory / "model.joblib", compress=3)
    (directory / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")


@classmethod
def load(cls, directory):
    # joblib/pickle is executable: load only trusted model artifacts.
    return joblib.load(Path(directory) / "model.joblib")