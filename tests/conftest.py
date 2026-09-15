import pytest

from re2 import embed, search


@pytest.fixture(autouse=True)
def reset_module_caches():
    """Module-level caches (query vectors, backoff, post maps, counts) must not leak between tests."""
    embed._query_down_until = 0.0
    embed._query_cache.clear()
    embed._indexes.clear()
    search._post_cache.clear()
    search._overall_cache.clear()
    yield
