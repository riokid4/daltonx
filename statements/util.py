"""Utility functions for statements, matches keys, and hashing."""
from hashlib import md5


__all__ = ['make_hash', 'mk_str', 'stmt_type']


def make_hash(s, n_bytes=14):
    """Make a signed integer hash from a matches key string.

    Parameters
    ----------
    s : str
        String to hash.
    n_bytes : int
        Number of bytes / hex digits to consider (default: 14 for shallow, 16 for full).

    Returns
    -------
    int
        A signed integer hash.
    """
    raw_h = int(md5(s.encode('utf-8')).hexdigest()[:n_bytes], 16)
    # Make it a signed int.
    return 16**n_bytes // 2 - raw_h


def mk_str(mk):
    """Format matches key string consistently for hashing."""
    return str(mk)


def stmt_type(obj, mk=True):
    """Return standardized object type String or class."""
    if hasattr(obj, '__class__') and mk:
        return type(obj)
    else:
        return type(obj).__name__
