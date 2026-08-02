"""
Co-occurrence expansion for catalog growth.

Mine which artists and tracks appear together in the same playlist/source group,
then generate new search queries for underrepresented but co-occurring artists,
growing the catalog organically from existing seeds.

Terminology
-----------
- context : a single source group (e.g. one playlist, one query group)
- pair    : two artists that share at least one common context
- score   : number of contexts in which a pair co-occurs
"""

import json
import os
import unicodedata
from collections import defaultdict
from itertools import combinations
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# source_ref values that identify a *playlist* context (not a loose query group)
_PLAYLIST_PREFIXES = ('config_playlist:', 'seed_playlist:', 'file:', 'exp_related',)

# source_ref values that should NOT be grouped (queries produce 1 track each)
_QUERY_PREFIXES = ('seed_query:', 'config_query:', 'curated_builtin',)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_artist(name: str) -> str:
    """Lower-case, strip accents, collapse whitespace."""
    if not isinstance(name, str):
        return ''
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = nfkd.encode('ascii', 'ignore').decode('ascii')
    return ' '.join(ascii_name.lower().split())


def _build_contexts(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return {context_key: [artist, ...]} from raw DataFrame.

    Each unique source_ref that looks like a playlist is its own context.
    For query-sourced tracks (which arrive one-per-query) we fall back to
    treating the *full dataset* as one weak global context so we still get
    some signal even on small catalogs.
    """
    contexts: dict[str, list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        artist = row.get('track_artist_name', '') or ''
        if not artist or str(artist).strip().lower() in ('', 'unknown artist'):
            continue

        source_ref = str(row.get('source_ref', '') or '')

        # Playlist contexts: use the ref as the context key
        is_playlist_ctx = any(source_ref.startswith(p) for p in _PLAYLIST_PREFIXES)
        is_query_ctx = any(source_ref.startswith(q) for q in _QUERY_PREFIXES)

        if is_playlist_ctx:
            ctx_key = source_ref
        elif is_query_ctx:
            # Loose queries → single global weak context
            ctx_key = '__global__'
        else:
            # NaN / unknown: use global
            ctx_key = '__global__'

        contexts[ctx_key].append(_normalise_artist(artist))

    return dict(contexts)


# ──────────────────────────────────────────────────────────────────────────────
# Core mining
# ──────────────────────────────────────────────────────────────────────────────

def mine_artist_cooccurrence(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """
    Compute pairwise artist co-occurrence scores from raw dataset.

    Returns
    -------
    dict mapping (artist_a, artist_b) -> co-occurrence count  (a < b lexically)
    """
    contexts = _build_contexts(df)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for ctx_key, artists in contexts.items():
        unique_in_ctx = sorted(set(artists))
        for a, b in combinations(unique_in_ctx, 2):
            pair = (a, b) if a < b else (b, a)
            pair_counts[pair] += 1

    return dict(pair_counts)


def artist_track_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return {normalised_artist: number_of_tracks} from the raw dataset."""
    counts: dict[str, int] = defaultdict(int)
    for name in df['track_artist_name'].dropna():
        key = _normalise_artist(str(name))
        if key:
            counts[key] += 1
    return dict(counts)


# ──────────────────────────────────────────────────────────────────────────────
# Query generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_cooccurrence_expansion_queries(
    df: pd.DataFrame,
    top_pairs: int = 40,
    queries_per_artist: int = 3,
    min_cooccurrence_score: int = 1,
    min_existing_tracks: int = 0,
    max_existing_tracks: int = 2,
) -> tuple[list[str], dict]:
    """
    Generate new search queries for underrepresented artists that co-occur
    frequently with well-represented artists.

    Parameters
    ----------
    df                   : raw track DataFrame
    top_pairs            : how many top co-occurrence pairs to consider
    queries_per_artist   : search queries generated per target artist
    min_cooccurrence_score : minimum times a pair must co-occur to be used
    min_existing_tracks  : target artist must have at least this many tracks
    max_existing_tracks  : target artist must have at most this many tracks
                           (use a low value to focus on underrepresented ones)

    Returns
    -------
    (queries, stats_dict)
    """
    if df.empty:
        return [], {}

    pairs = mine_artist_cooccurrence(df)
    if not pairs:
        logger.warning("No co-occurrence pairs found. Catalog may be too small or lacks source_ref.")
        return [], {}

    track_counts = artist_track_counts(df)

    # Sort by co-occurrence score descending
    sorted_pairs = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)

    # All raw artist names (for display in queries – prefer original capitalisation)
    orig_names: dict[str, str] = {}
    for name in df['track_artist_name'].dropna():
        key = _normalise_artist(str(name))
        if key and key not in orig_names:
            orig_names[key] = str(name).strip()

    queries: list[str] = []
    target_artists_seen: set[str] = set()
    stats = {
        'pairs_considered': 0,
        'pairs_used': 0,
        'target_artists': [],
    }

    for (artist_a, artist_b), score in sorted_pairs[:top_pairs * 4]:
        if score < min_cooccurrence_score:
            break

        stats['pairs_considered'] += 1

        for target, anchor in ((artist_b, artist_a), (artist_a, artist_b)):
            if target in target_artists_seen:
                continue

            count = track_counts.get(target, 0)
            if not (min_existing_tracks <= count <= max_existing_tracks):
                continue

            display_name = orig_names.get(target, target)
            anchor_name = orig_names.get(anchor, anchor)

            # Generate varied query styles
            new_queries = [
                f"{display_name} best songs",
                f"{display_name} top hits",
                f"{display_name} {anchor_name} mix",
            ][:queries_per_artist]

            queries.extend(new_queries)
            target_artists_seen.add(target)
            stats['target_artists'].append(target)
            stats['pairs_used'] += 1

            if len(target_artists_seen) >= top_pairs:
                break

        if len(target_artists_seen) >= top_pairs:
            break

    # Deduplicate while preserving order
    seen_q: set[str] = set()
    deduped = []
    for q in queries:
        if q not in seen_q:
            seen_q.add(q)
            deduped.append(q)

    logger.info(
        f"Co-occurrence expansion: {len(deduped)} queries for "
        f"{len(target_artists_seen)} target artists "
        f"({stats['pairs_used']}/{stats['pairs_considered']} pairs used)"
    )
    return deduped, stats


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_cooccurrence_graph(pairs: dict[tuple[str, str], int], path: str) -> None:
    """Persist co-occurrence graph to JSON (keys are 'a|||b' strings)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serialisable = {f"{a}|||{b}": count for (a, b), count in pairs.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved co-occurrence graph ({len(serialisable)} pairs) to {path}")


def load_cooccurrence_graph(path: str) -> dict[tuple[str, str], int]:
    """Load a previously saved co-occurrence graph."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return {tuple(k.split('|||', 1)): int(v) for k, v in raw.items() if '|||' in k}
    except Exception as e:
        logger.warning(f"Could not load co-occurrence graph from {path}: {e}")
        return {}


def merge_cooccurrence_graphs(
    existing: dict[tuple[str, str], int],
    new: dict[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    """Merge two co-occurrence graphs by summing counts."""
    merged = dict(existing)
    for pair, count in new.items():
        merged[pair] = merged.get(pair, 0) + count
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def top_pairs_report(pairs: dict[tuple[str, str], int], n: int = 20) -> str:
    """Return a human-readable summary of top co-occurrence pairs."""
    sorted_pairs = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:n]
    lines = [f"{'Artist A':<30} {'Artist B':<30} {'Score':>6}"]
    lines.append('-' * 68)
    for (a, b), score in sorted_pairs:
        lines.append(f"{a:<30} {b:<30} {score:>6}")
    return '\n'.join(lines)
