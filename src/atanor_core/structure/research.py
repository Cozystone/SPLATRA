"""Look the object up before building it.

Asking an LLM what a car is made of works because cars are in its weights. Ask it
about a trebuchet counterweight sling, a tokamak divertor or the sternocleidomastoid
and the answer degrades into plausible-sounding invention — which is exactly the
failure mode that makes a generated "model" untrustworthy.

So the structural pass gets evidence first. We fetch the encyclopedia article for
the object and pull out the sections that describe how it is built — Anatomy,
Structure, Components, Parts, Design, Construction — and hand that text to the
decomposer as grounding. An unfamiliar object stops being a guess and becomes a
reading comprehension problem, which small models are far better at.

What is retrieved is cached on disk, so each object is researched once and the
knowledge accumulates: the second time SPLATRA meets a trebuchet it already knows.

Honest scope: this grounds structure in TEXT. Retrieving an anatomical *diagram*
and reading it would need a vision-language model, which is not wired up here — the
local LLM is text-only. Section text is the part we can actually use today.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import time
import urllib.request
from typing import Any, Dict, List, Optional

_UA = "SPLATRA/0.1 (3D structural research; contact: local)"
_API = "https://en.wikipedia.org/w/api.php"

# Sections that actually describe how a thing is put together.
_WANTED = ("anatomy", "structure", "components", "component", "parts", "part",
           "design", "construction", "mechanism", "layout", "morphology",
           "body plan", "architecture", "assembly", "organs", "systems",
           "physiology", "characteristics", "description", "features")

_CACHE_DIR = os.environ.get(
    "SPLATRA_KNOWLEDGE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "out", "knowledge"))


def _get(params: Dict[str, str], timeout: float = 15.0) -> Dict[str, Any]:
    # A burst of section lookups gets throttled, and a swallowed 429 looks exactly
    # like "this object does not exist" — so back off and retry rather than
    # silently reporting ignorance.
    url = _API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(0.6 * (attempt + 1))
    raise last if last else RuntimeError("lookup failed")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:60] or "object"


def _best_title(query: str, timeout: float) -> Optional[str]:
    data = _get({"action": "query", "list": "search", "srsearch": query,
                 "srlimit": "1"}, timeout)
    hits = (data.get("query", {}) or {}).get("search", [])
    return hits[0]["title"] if hits else None


def _sections(title: str, timeout: float) -> str:
    """Concatenate the build-describing sections of the article."""
    data = _get({"action": "parse", "page": title, "prop": "sections"}, timeout)
    secs = (data.get("parse", {}) or {}).get("sections", [])
    keep = [s for s in secs
            if any(w in (s.get("line") or "").lower() for w in _WANTED)][:4]
    out: List[str] = []
    for s in keep:
        try:
            d = _get({"action": "query", "prop": "extracts", "explaintext": "1",
                      "titles": title, "redirects": "1"}, timeout)
            page = list((d.get("query", {}) or {}).get("pages", {}).values())[0]
            text = page.get("extract") or ""
            line = s.get("line") or ""
            m = re.search(r"^=+\s*" + re.escape(line) + r"\s*=+\s*$(.*?)(?=^=|\Z)",
                          text, re.S | re.M)
            if m:
                out.append(line + ": " + re.sub(r"\s+", " ", m.group(1)).strip()[:900])
        except Exception:
            continue
        if len(out) >= 3:
            break
    return "\n".join(out)


def research(topic: str, timeout: float = 15.0,
             use_cache: bool = True) -> Dict[str, Any]:
    """topic -> {title, summary, detail, source, cached}. Never raises."""
    key = _slug(topic)
    path = os.path.join(_CACHE_DIR, key + ".json")
    if use_cache and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                got = json.load(f)
            got["cached"] = True
            return got
        except Exception:
            pass

    out: Dict[str, Any] = {"title": None, "summary": "", "detail": "",
                           "source": None, "cached": False}
    try:
        title = _best_title(topic, timeout)
        if not title:
            return out
        data = _get({"action": "query", "prop": "extracts", "exintro": "1",
                     "explaintext": "1", "redirects": "1", "titles": title}, timeout)
        page = list((data.get("query", {}) or {}).get("pages", {}).values())[0]
        out["title"] = page.get("title") or title
        out["summary"] = re.sub(r"\s+", " ", (page.get("extract") or ""))[:1200]
        out["source"] = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
            (out["title"] or "").replace(" ", "_"))
        try:
            out["detail"] = _sections(out["title"], timeout)
        except Exception:
            out["detail"] = ""
    except Exception:
        return out

    if out["summary"]:
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
        except Exception:
            pass
    return out


def evidence(topic: str, timeout: float = 15.0) -> str:
    """A compact block of retrieved facts to ground the decomposer, or ""."""
    r = research(topic, timeout=timeout)
    if not r.get("summary"):
        return ""
    parts = ["REFERENCE on " + str(r.get("title"))]
    parts.append(r["summary"][:800])
    if r.get("detail"):
        parts.append(r["detail"][:1400])
    return "\n".join(parts)
