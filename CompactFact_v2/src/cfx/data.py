import json, os, random
from .text import split_of
ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]

def splits(seed=42, golden_size=150):
    train = load_jsonl(os.path.join(ROOT, 'train.jsonl'))
    val = load_jsonl(os.path.join(ROOT, 'val.jsonl'))
    dev = [c for c in train if split_of(c, seed) == 'dev']
    rng = random.Random(seed)
    golden = rng.sample(dev, golden_size)
    gid = {c['call_id'] for c in golden}
    rest = [c for c in dev if c['call_id'] not in gid]
    # dev_A — обучение селектора текущего значения; dev_B — перепроверка элиты эволюции
    return {'train': train, 'val': val, 'golden': golden, 'dev_rest': rest, 'dev_A': rest[:250], 'dev_B': rest[250:]}

def catalog():
    return json.load(open(os.path.join(ROOT, 'catalog.json'), encoding='utf-8'))
