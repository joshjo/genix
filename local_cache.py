import json
import rocksdb


_cache = rocksdb.DB('dbs/cache.db', rocksdb.Options(create_if_missing=True))


def set(key, value):
    _cache.put(str.encode(key), json.dumps(value).encode())


def get(key):
    value = _cache.get(str.encode(key))
    if not value:
        return
    return json.loads(value)
