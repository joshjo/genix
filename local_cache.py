import hashlib
import json
from rocksdict import Rdict


def hash_list(list_data):
    """Return a hash from the given data"""
    data = sorted(list_data)
    return hashlib.sha1(json.dumps(data).encode("utf-8")).hexdigest()


def get_hash_key(gen):
    gen_str = "".join(str(i) for i in gen)
    return gen_str


class LocalCache:
    _caches = {}

    def __init__(self, benchmark_queries):
        hash_name = hash_list(benchmark_queries)
        self._cache = Rdict(f"dbs/cache/{hash_name}.db")

    def put(self, key, value):
        self._cache[key] = value

    def get(self, key):
        try:
            return self._cache[key]
        except:
            return

    def get_all(self):
        return dict(self._cache)

    def batch_update(self, new_values):
        for key, value in new_values.items():
            self._cache[key] = value
