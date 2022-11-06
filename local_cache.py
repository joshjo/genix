import json
from rocksdict import Rdict


_cache = Rdict('dbs/cache.db')


def put(key, value):
    _cache[key] = value


def get(key):
    try:
        return _cache[key]
    except:
        return
