"""
ticker_storage.py – Persistent storage for ticker overrides using GitHub API.

On Streamlit Cloud, local files reset when the app sleeps. This module
reads/writes ticker_overrides.json directly to the GitHub repo so
custom tickers survive restarts.

Requires GITHUB_TOKEN in Streamlit secrets (or environment variable).
Falls back to local file if GitHub is unavailable.
"""

import json
import logging
import os
from base64 import b64decode, b64encode
from typing import Optional

import requests
import streamlit as st

logger = logging.getLogger(__name__)

REPO = "aicodepathways/Matt-Michael"
FILE_PATH = "ticker_overrides.json"
BRANCH = "main"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "ticker_overrides.json")

DEFAULT_OVERRIDES = {
    "added_asx": [],
    "added_global": [],
    "added_commodities": {},
    "removed": [],
}


def _get_token() -> Optional[str]:
    """Get GitHub token from Streamlit secrets or environment."""
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        return os.environ.get("GITHUB_TOKEN")


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def load_overrides() -> dict:
    """
    Load ticker overrides. Tries GitHub first, falls back to local file.
    Caches in session state to avoid repeated API calls.
    """
    # Check session state cache first
    if "ticker_overrides" in st.session_state:
        return st.session_state["ticker_overrides"]

    token = _get_token()

    # Try GitHub API
    if token:
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}",
                headers=_github_headers(token),
                params={"ref": BRANCH},
                timeout=10,
            )
            if resp.status_code == 200:
                content = b64decode(resp.json()["content"]).decode("utf-8")
                overrides = json.loads(content)
                # Cache and also write local copy
                st.session_state["ticker_overrides"] = overrides
                _write_local(overrides)
                logger.info("Loaded ticker overrides from GitHub.")
                return overrides
            elif resp.status_code == 404:
                # File doesn't exist in repo yet — use defaults
                logger.info("No ticker_overrides.json in repo, using defaults.")
                st.session_state["ticker_overrides"] = DEFAULT_OVERRIDES.copy()
                return DEFAULT_OVERRIDES.copy()
            else:
                logger.warning("GitHub API returned %d, falling back to local.", resp.status_code)
        except Exception as e:
            logger.warning("GitHub API failed: %s, falling back to local.", e)

    # Fall back to local file
    if os.path.exists(LOCAL_PATH):
        try:
            with open(LOCAL_PATH) as f:
                overrides = json.load(f)
            st.session_state["ticker_overrides"] = overrides
            return overrides
        except (json.JSONDecodeError, IOError):
            pass

    return DEFAULT_OVERRIDES.copy()


def save_overrides(overrides: dict):
    """
    Save ticker overrides. Writes to GitHub and local file.
    """
    # Update session state
    st.session_state["ticker_overrides"] = overrides

    # Write local
    _write_local(overrides)

    # Push to GitHub
    token = _get_token()
    if token:
        try:
            _push_to_github(overrides, token)
            logger.info("Saved ticker overrides to GitHub.")
        except Exception as e:
            logger.warning("Failed to push to GitHub: %s. Local save succeeded.", e)


def _write_local(overrides: dict):
    try:
        with open(LOCAL_PATH, "w") as f:
            json.dump(overrides, f, indent=2)
    except IOError as e:
        logger.warning("Failed to write local overrides: %s", e)


def _push_to_github(overrides: dict, token: str):
    """Push ticker_overrides.json to the GitHub repo."""
    headers = _github_headers(token)
    content = json.dumps(overrides, indent=2)
    encoded = b64encode(content.encode("utf-8")).decode("utf-8")

    # Get current file SHA (needed for update, not for create)
    sha = None
    resp = requests.get(
        f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}",
        headers=headers,
        params={"ref": BRANCH},
        timeout=10,
    )
    if resp.status_code == 200:
        sha = resp.json()["sha"]

    # Create or update
    payload = {
        "message": "Update ticker overrides from dashboard",
        "content": encoded,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(
        f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}",
        headers=headers,
        json=payload,
        timeout=10,
    )

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"GitHub API PUT failed: {resp.status_code} {resp.text[:200]}")
