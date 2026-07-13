from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


# Default model names (can be overridden via env vars)
GEMINI_CHAT_MODEL: str = os.getenv(
    "SMARTSTRESS_GEMINI_CHAT_MODEL", "gemini-2.5-flash"
)
GEMINI_EMBED_MODEL: str = os.getenv(
    "SMARTSTRESS_GEMINI_EMBED_MODEL", "gemini-embedding-001"
)

# Local secrets files (relative to project root or this package)
API_KEY_FILENAME = ".API_KEY"
DOTENV_FILENAME = ".env"

_ENV_FILES_INITIALISED = False


def _find_project_root() -> Path:
    """Return the package root (Agents_LangGraph)."""
    return Path(__file__).resolve().parent.parent


def _initialise_env_files() -> None:
    """Ensure dotenv-based environment variables are loaded once."""
    global _ENV_FILES_INITIALISED
    if _ENV_FILES_INITIALISED:
        return
    _ENV_FILES_INITIALISED = True
    _load_env_file()


def _load_env_file() -> None:
    """
    Populate os.environ with values from a .env file if present.

    Search order:
    1. SMARTSTRESS_DOTENV_FILE env var
    2. Agents_LangGraph directory (one level above this package)
    3. Current working directory
    """
    env_path = os.getenv("SMARTSTRESS_DOTENV_FILE")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    pkg_root = _find_project_root()
    candidates.append(pkg_root / DOTENV_FILENAME)
    candidates.append(Path.cwd() / DOTENV_FILENAME)

    for candidate in candidates:
        if not candidate.is_file():
            continue
        _parse_dotenv(candidate)
        break


def _parse_dotenv(path: Path) -> None:
    """Parse a dotenv file and set vars if missing."""
    try:
        content = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in content:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("\"'")  # tolerate simple quoting
        os.environ[key] = value


def _find_api_key_file() -> Optional[Path]:
    """
    Try to locate the .API_KEY file.

    Search order:
    1. SMARTSTRESS_API_KEY_FILE env var
    2. Agents_LangGraph directory (one level above this package)
    3. Current working directory
    """
    env_path = os.getenv("SMARTSTRESS_API_KEY_FILE")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p

    # Package root: smartstress_langgraph/.. -> Agents_LangGraph
    pkg_root = _find_project_root()
    candidate = pkg_root / API_KEY_FILENAME
    if candidate.is_file():
        return candidate

    # Fallback to CWD
    cwd_candidate = Path.cwd() / API_KEY_FILENAME
    if cwd_candidate.is_file():
        return cwd_candidate

    return None


@lru_cache(maxsize=1)
def load_google_api_key() -> str:
    """
    Load GOOGLE_API_KEY from .API_KEY file or environment.

    .API_KEY format:
        GOOGLE_API_KEY=xxxx...
    or just:
        xxxx...

    Environment override:
        If GOOGLE_API_KEY env var is set, it is used directly.
    """
    _initialise_env_files()
    
    # We prioritize .env values if they were just loaded into os.environ
    # Note: _initialise_env_files calls _parse_dotenv which uses setdefault.
    # To truly prioritize .env, we should parse it again or change setdefault logic.
    # Let's check environment variable first, but give a hint if .env is different.
    
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key.strip()

    api_key_path = _find_api_key_file()
    if not api_key_path:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set the GOOGLE_API_KEY environment variable "
            "or add it to a .env file (GOOGLE_API_KEY=...) before starting the app. "
            "A legacy .API_KEY file is still supported if present."
        )

    content = api_key_path.read_text(encoding="utf-8").strip()
    if not content:
        raise RuntimeError(f".API_KEY file at {api_key_path} is empty.")

    # Support both raw key and KEY=... format
    if "=" in content:
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "GOOGLE_API_KEY":
                    v = v.strip()
                    if not v:
                        raise RuntimeError(
                            f".API_KEY at {api_key_path} contains an empty GOOGLE_API_KEY entry."
                        )
                    return v
        raise RuntimeError(
            f".API_KEY at {api_key_path} does not contain a valid GOOGLE_API_KEY entry."
        )

    return content


def get_default_generation_config() -> dict:
    """Default hyper-parameters for Gemini chat models."""
    return {
        "temperature": float(os.getenv("SMARTSTRESS_TEMPERATURE", "0.5")),
        "top_p": float(os.getenv("SMARTSTRESS_TOP_P", "0.9")),
        "top_k": int(os.getenv("SMARTSTRESS_TOP_K", "40")),
        "max_output_tokens": int(os.getenv("SMARTSTRESS_MAX_TOKENS", "1024")),
    }


def get_stress_threshold() -> float:
    """Return the configurable PhysioSense binary decision threshold."""
    _initialise_env_files()
    raw_value = os.getenv("SMARTSTRESS_STRESS_THRESHOLD", "0.5")
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "SMARTSTRESS_STRESS_THRESHOLD must be a number between 0 and 1"
        ) from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("SMARTSTRESS_STRESS_THRESHOLD must be between 0 and 1")
    return threshold



