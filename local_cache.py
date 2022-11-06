import hashlib
import json
from rocksdict import Rdict


def hash_list(list_data):
    """Return a hash from the given data"""
    data = sorted(list_data)
    return hashlib.sha1(json.dumps(data).encode("utf-8")).hexdigest()


class LocalCache:
    _caches = {}

    def __new__(cls, benchmark_queries):
        hash_name = hash_list(benchmark_queries)
        if hash_name not in cls._caches:
            new_cache = Rdict(f"dbs/cache/{hash_name}.db")
            cls._caches[hash_name] = new_cache
        return cls._caches[hash_name]

    def __init__(self, benchmark_queries):
        self._cache = self.__new__(benchmark_queries)


    def put(key, value):
        self._cache[key] = value


    def get(key):
        try:
            return self._cache[key]
        except:
            return
