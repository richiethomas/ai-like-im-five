"""Shared test config: load API keys from .env for smoke tests."""

from pathlib import Path

from dotenv import load_dotenv

# Repo root .env — smoke tests need real keys; unit tests don't care.
load_dotenv(Path(__file__).parent.parent / ".env")

FIXTURE_CACHE = Path(__file__).parent / "fixtures" / "cache"
FIXTURE_CACHE.mkdir(parents=True, exist_ok=True)
