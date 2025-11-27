import hashlib
import json
from collections import OrderedDict
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CacheStatistics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


    @property
    def total_requests(self) -> int:
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.hits/ self.total_requests) * 100
    
    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0

class Text2CypherCache:

    def __init__(self, max_size: int = 128):

        self.max_size = max_size
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.stats = CacheStatistics()

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _make_cache_key(self, question: str, schema: str) -> str:

        q_hash = self._hash_content(question.lower().strip())
        s_hash = self._hash_content(schema)
        return f"{q_hash}:{s_hash}"
    
    def get(self, question: str, schema:str) -> Optional[str]:

        key = self._make_cache_key(question, schema)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats.hits +=1
            return self.cache[key]
        
        self.stats.misses +=1
        return None
    
    def put(self, question:str, schema:str, query:str) -> None:

        key = self._make_cache_key(question, schema)

        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = query

        if len(self.cache) > self.max_size:
            self.cache.popitem(last= False) 
            self.stats.evictions +=1

    def clear(self) -> None:
        self.cache.clear()
        self.stats.reset()
    
    def get_stats(self) -> CacheStatistics:
        return self.stats
    
    def print_stats(self) -> None:

        print("\n" + "=" * 80)
        print("Text2Cypher cache statistics")
        print("=" * 80)
        print(f"Cache size:        {len(self.cache)}/{self.max_size}")
        print(f"Total requests:    {self.stats.total_requests}")
        print(f"Cache hits:        {self.stats.hits}")
        print(f"Cache misses:      {self.stats.misses}")
        print(f"Hit rate:          {self.stats.hit_rate:.2f}%")
        print(f"Evictions:         {self.stats.evictions}")
        print("=" * 80 + "\n")
    
    def __len__(self) -> int:
        return len(self.cache)





        