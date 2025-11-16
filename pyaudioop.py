# pyaudioop.py — place at project root to satisfy `import pyaudioop`
# This re-exports functions from audioop (if available) or from audioop-lts.
try:
    # first try the real audioop (works if audioop-lts or stdlib audioop is present)
    from audioop import *
except Exception:
    try:
        # try audioop-lts if it installs under a different name (rare)
        from audioop_lts import *
    except Exception:
        # helpful error if nothing is present
        raise ImportError(
            "Neither 'audioop' nor 'audioop-lts' are available. "
            "Install 'audioop-lts' in requirements.txt or run on Python <= 3.12."
        )
