"""Verify TLS using native trust roots without modifying global SSL behavior."""

import os
import ssl
from functools import lru_cache

import truststore


@lru_cache(maxsize=1)
def verified_context() -> ssl.SSLContext:
    # Preserve explicitly configured enterprise CA bundles.
    if os.getenv("SSL_CERT_FILE") or os.getenv("SSL_CERT_DIR"):
        return ssl.create_default_context(
            cafile=os.getenv("SSL_CERT_FILE"), capath=os.getenv("SSL_CERT_DIR")
        )
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
