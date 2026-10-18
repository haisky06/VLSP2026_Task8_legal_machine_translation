"""Semantic, monotone EN/VI legal-document alignment; Python + NumPy only.

This is an independent implementation, inspired by the anchored many-to-many
alignment design described at https://github.com/bfsujason/bertalign (Liu & Zhu),
and the document-integrity/unaligned-content reporting principles of DocHPLT:
https://aclanthology.org/2025.wmt-1.17/ . No Bertalign code is copied.

The caller supplies ``embedder(list_of_texts)``. It must return multilingual,
translation-retrieval embeddings and must handle long inputs without silently
truncating them. LaBSE with tokenizer-aware subsegment pooling is suitable.
Cosine similarity is NOT a calibrated probability or measured alignment precision.
"""

from bisect import bisect_left
from collections import Counter
import math
import re
import unicodedata

import numpy as np


ALIGNMENT_VERSION = "qwen_legal_anchors_semantic_dp_v6.0"
_PREFIX = r"^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*"
_HEADINGS = {
    "en": [("annex", r"(?:ANNEX|APPENDIX)"), ("part", r"PART"),
           ("chapter", r"CHAPTER"), ("section", r"SECTION"), ("article", r"ARTICLE")],
    "vi": [("annex", r"PHỤ\s*LỤC"), ("part", r"PHẦN"),
           ("chapter", r"CHƯƠNG"), ("section", r"MỤC"), ("article", r"ĐIỀU")],
}
_HEADING_PATTERNS = {
    lang: [(kind, re.compile(_PREFIX + word + r"\s+([0-9]+[A-Za-z]?|[IVXLCDM]+|[A-Z])(?=\b|[.:(])", re.I))
           for kind, word in entries] for lang, entries in _HEADINGS.items()
}
_MONTHS = dict(zip("january february march april may june july august september october november december".split(), range(1, 13)))
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.I)
_ABBREVIATIONS = {"art", "arts", "no", "nos", "mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "tp", "qđ", "nđ"}


def _nonspace_len(text):
    return sum(not c.isspace() for c in text)


def _marker(value):
    value = value.upper().strip()
    if value.isdigit():
        return str(int(value))
    if value and all(c in "IVXLCDM" for c in value):
        table = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        total = previous = 0
        for char in reversed(value):
            current = table[char]
            total += -current if current < previous else current
            previous = max(previous, current)
        return str(total)
    return value


_CLAUSE_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?(\d{1,3})(?:[./)]|\s*[-–])\s+")
_POINT_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?([a-zđ])(?:[.)/])\s+", re.I)

def _heading(line, lang):
    for kind, pattern in _HEADING_PATTERNS[lang]:
        found = pattern.match(line)
        if found: return kind, _marker(found.group(1))
    clause = _CLAUSE_RE.match(line)
    if clause: return "clause", _marker(clause.group(1))
    point = _POINT_RE.match(line)
    if point: return "point", point.group(1).lower()
    return None


def legal_atomic_spans(text, lang, max_unit_chars=900):
    """Return a contiguous exact partition, including whitespace and Markdown.

    Paragraphs, sentence endings and semicolons split *inside* articles too.
    Long unpunctuated text is split on whitespace, then by character as a last
    resort. Neither normalization nor filtering changes the returned raw spans.
    """
    if lang not in _HEADING_PATTERNS:
        raise ValueError("lang must be 'en' or 'vi'")
    if max_unit_chars < 80:
        raise ValueError("max_unit_chars must be >= 80")
    text = str(text)
    if not text:
        return []
    boundaries = {0, len(text)}
    for line in re.finditer(r"[^\r\n]*(?:\r\n|\n|\r|$)", text):
        if line.end() == line.start():
            continue
        raw = line.group(0)
        if raw.strip():
            boundaries.add(line.start())
            boundaries.add(line.end())
            if not _heading(raw, lang):
                for match in re.finditer(r"[.!?;][\"'”’)]*\s+", raw):
                    left = raw[:match.start() + 1]
                    word = re.search(r"([\w.]+)\.$", left)
                    if word and (word.group(1).lower() in _ABBREVIATIONS or
                                 len(word.group(1)) == 1 or word.group(1).isdigit()):
                        continue
                    boundaries.add(line.start() + match.end())
    initial = sorted(boundaries)
    # Merge whitespace-only pieces into the previous span (or the first span).
    compact = [0]
    for start, end in zip(initial, initial[1:]):
        if text[start:end].strip():
            if start and text[compact[-1]:start].strip():
                compact.append(start)
    compact.append(len(text))
    final = [0]
    for start, end in zip(compact, compact[1:]):
        cursor = start
        while end - cursor > max_unit_chars:
            cap = cursor + max_unit_chars
            spaces = [m.end() for m in re.finditer(r"\s+", text[cursor:cap])
                      if m.end() >= max_unit_chars // 2]
            cut = cursor + spaces[-1] if spaces else cap
            final.append(cut)
            cursor = cut
        final.append(end)
    final = sorted(set(final))
    units = []
    scope = {"annex": "", "part": "", "chapter": "", "section": "", "article": "", "clause": "", "point": ""}
    occurrences = Counter()
    for start, end in zip(final, final[1:]):
        raw = text[start:end]
        first = next((line for line in raw.splitlines() if line.strip()), "")
        heading = _heading(first, lang)
        # A forced character split must not reinterpret a continuation as a heading.
        starts_line = start == 0 or text[start - 1] in "\r\n" or not text[:start].strip()
        if not starts_line:
            heading = None
        anchor_key = None
        if heading:
            kind, number = heading
            if kind == "annex":
                scope.update(annex=number, part="", chapter="", section="", article="", clause="", point="")
            elif kind == "part":
                scope.update(part=number, chapter="", section="", article="", clause="", point="")
            elif kind == "chapter":
                scope.update(chapter=number, section="", article="", clause="", point="")
            elif kind == "section":
                scope.update(section=number, article="", clause="", point="")
            elif kind == "article":
                scope.update(article=number, clause="", point="")
            elif kind == "clause":
                scope.update(clause=number, point="")
            else:
                scope["point"] = number
            # Qwen-style hierarchy: same marker is only anchored within its parent.
            parents = {"annex": (), "part": (scope["annex"],),
                       "chapter": (scope["annex"], scope["part"]),
                       "section": (scope["annex"], scope["part"], scope["chapter"]),
                       "article": (scope["annex"], scope["part"], scope["chapter"], scope["section"]),
                       "clause": (scope["annex"], scope["part"], scope["chapter"], scope["section"], scope["article"]),
                       "point": (scope["annex"], scope["part"], scope["chapter"], scope["section"], scope["article"], scope["clause"])}
            anchor_key = (kind, *parents[kind], number)
            occurrences[anchor_key] += 1
        units.append({"start": start, "end": end, "text": raw, "heading": heading,
                      "anchor_key": anchor_key, "scope": dict(scope),
                      "occurrence": occurrences[anchor_key] if anchor_key else None,
                      "chars": _nonspace_len(raw)})
    assert "".join(unit["text"] for unit in units) == text
    return units


def _numeric_items(text):
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _MONTH_RE.sub(lambda match: str(_MONTHS[match.group(0).lower()]), normalized)
    # Month names / leading zeroes no longer create a categorical mismatch.
    # Normalize only conventional 3-digit thousands groups; retain decimals.
    normalized = re.sub(r"\b\d{1,3}(?:[.,]\d{3})+(?!\d)",
                        lambda match: re.sub(r"[.,]", "", match.group(0)), normalized)
    return Counter(str(int(number)) for number in re.findall(r"\d+", normalized))


def _number_penalty(left, right):
    if not left and not right:
        return 0.0
    def weight(number):
        return 0.40 if int(number) <= 31 else 1.0
    common = sum(weight(k) * v for k, v in (left & right).items())
    total = sum(weight(k) * v for nums in (left, right) for k, v in nums.items())
    disagreement = 1.0 - 2.0 * common / max(total, 1e-9)
    # A soft signal; textual dates and translations with differing enumerations
    # must not discard otherwise good entire paragraphs.
    return 0.16 * disagreement


def _unit_groups(units, max_group):
    groups = {}
    for start in range(len(units)):
        for size in range(1, min(max_group, len(units) - start) + 1):
            members = units[start:start + size]
            if any(member["heading"] for member in members[1:]):
                break  # Never merge across a detected major legal boundary.
            text = "".join(member["text"] for member in members)
            chars = sum(member["chars"] for member in members)
            if chars:
                groups[start, size] = {"text": text, "chars": chars,
                                       "numbers": _numeric_items(text),
                                       "heading": members[0]["heading"]}
    return groups


def _embedding_matrix(embedder, texts):
    matrix = np.asarray(embedder(texts), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError("embedder must return an [n_texts, embedding_dim] array")
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(lengths < 1e-8):
        raise ValueError("embedder returned NaN/Inf/zero embeddings")
    return matrix / lengths


def _monotone_chain(candidates):
    """Longest strictly increasing chain; every candidate has a unique EN index."""
    candidates = sorted(candidates)
    tails, tail_indices, previous = [], [], [-1] * len(candidates)
    for index, candidate in enumerate(candidates):
        position = bisect_left(tails, candidate[1])
        if position:
            previous[index] = tail_indices[position - 1]
        if position == len(tails):
            tails.append(candidate[1]); tail_indices.append(index)
        else:
            tails[position] = candidate[1]; tail_indices[position] = index
    result = []
    cursor = tail_indices[-1] if tail_indices else -1
    while cursor >= 0:
        result.append(candidates[cursor]); cursor = previous[cursor]
    return list(reversed(result))


def _heading_anchors(en_units, vi_units, en_groups, vi_groups, threshold):
    by_key_en, by_key_vi = {}, {}
    for units, output in ((en_units, by_key_en), (vi_units, by_key_vi)):
        for index, unit in enumerate(units):
            if unit["anchor_key"]:
                output.setdefault(unit["anchor_key"], []).append(index)
    candidates = []
    for key in by_key_en.keys() & by_key_vi.keys():
        source_ids, target_ids = by_key_en[key], by_key_vi[key]
        left = np.stack([en_groups[i, 1]["embedding"] for i in source_ids])
        right = np.stack([vi_groups[j, 1]["embedding"] for j in target_ids])
        similarities = left @ right.T
        # Repeated Article 1 etc. are NOT zipped by occurrence. Require a mutual
        # semantic best match with a margin; otherwise leave matching to the DP.
        for local_i, source_i in enumerate(source_ids):
            order = np.argsort(similarities[local_i])[::-1]
            local_j = int(order[0]); score = float(similarities[local_i, local_j])
            if score < max(0.56, threshold - 0.04) or int(np.argmax(similarities[:, local_j])) != local_i:
                continue
            row_margin = score - float(similarities[local_i, order[1]]) if len(order) > 1 else 1.0
            column = np.sort(similarities[:, local_j])
            column_margin = score - float(column[-2]) if len(column) > 1 else 1.0
            if min(row_margin, column_margin) >= 0.035:
                candidates.append((source_i, target_ids[local_j]))
    return _monotone_chain(candidates)


def _semantic_anchors(en_groups, vi_groups, en_start, en_end, vi_start, vi_end, threshold):
    """Memory-bounded mutual nearest-neighbor anchors for large unheaded spans."""
    n, m = en_end - en_start, vi_end - vi_start
    if not n or not m:
        return []
    left = np.stack([en_groups[i, 1]["embedding"] for i in range(en_start, en_end)])
    right = np.stack([vi_groups[j, 1]["embedding"] for j in range(vi_start, vi_end)])
    row_best = np.full(n, -1, dtype=np.int32)
    row_value = np.full(n, -np.inf, dtype=np.float32)
    row_second = np.full(n, -np.inf, dtype=np.float32)
    col_best = np.full(m, -1, dtype=np.int32)
    col_value = np.full(m, -np.inf, dtype=np.float32)
    col_second = np.full(m, -np.inf, dtype=np.float32)
    for start in range(0, n, 128):
        similarities = left[start:start + 128] @ right.T
        count = similarities.shape[0]
        local_best = np.argmax(similarities, axis=1)
        row_best[start:start + count] = local_best
        row_value[start:start + count] = similarities[np.arange(count), local_best]
        if m > 1:
            row_second[start:start + count] = np.partition(similarities, -2, axis=1)[:, -2]
        for local_i in range(count):
            values = similarities[local_i]
            better = values > col_value
            col_second = np.where(better, col_value, np.maximum(col_second, values))
            col_best = np.where(better, start + local_i, col_best)
            col_value = np.maximum(col_value, values)
    candidates = []
    for i, j in enumerate(row_best):
        value = float(row_value[i])
        if (int(col_best[j]) == i and value >= max(0.83, threshold + 0.12)
                and value - float(row_second[i]) >= 0.055
                and value - float(col_second[j]) >= 0.055):
            candidates.append((en_start + i, vi_start + int(j)))
    return _monotone_chain(candidates)


def _candidate_features(en_groups, vi_groups, i, j, a, b, ratio, threshold):
    left, right = en_groups.get((i, a)), vi_groups.get((j, b))
    if left is None or right is None:
        return None
    first, second = left["heading"], right["heading"]
    if first and second and first[0] == second[0] and first[1] != second[1]:
        return None
    cosine = float(np.dot(left["embedding"], right["embedding"]))
    length_ratio = right["chars"] / left["chars"]
    if not 0.08 <= length_ratio <= 12.0:
        return None
    numeric_penalty = _number_penalty(left["numbers"], right["numbers"])
    length_penalty = min(0.15, 0.035 * abs(math.log(length_ratio / ratio)))
    structure_bonus = 0.025 if first and first == second else 0.0
    confidence = min(1.0, cosine - numeric_penalty - length_penalty + structure_bonus)
    if confidence < threshold:
        return None
    # Do not hide unrelated added content inside a high-similarity merged pair.
    # Each member needs semantic support from a member or the opposing group.
    if a + b > 2:
        left_members = np.stack([en_groups[k, 1]["embedding"] for k in range(i, i + a)])
        right_members = np.stack([vi_groups[k, 1]["embedding"] for k in range(j, j + b)])
        member_sim = left_members @ right_members.T
        left_support = np.maximum(member_sim.max(axis=1), left_members @ right["embedding"])
        right_support = np.maximum(member_sim.max(axis=0), right_members @ left["embedding"])
        if min(float(left_support.min()), float(right_support.min())) < max(0.36, threshold - 0.25):
            return None
    return cosine, confidence, numeric_penalty


def _region_dp(en_groups, vi_groups, es, ee, vs, ve, ratio, threshold, max_group, max_dp_cells):
    n, m = ee - es, ve - vs
    if not n or not m:
        return [], {"en_units": n, "vi_units": m, "banded": False, "states": 0}
    banded = (n + 1) * (m + 1) > max_dp_cells
    width = max(4, min(96, max_dp_cells // max(2 * (n + 1), 1)))
    def bounds(i):
        if not banded:
            return 0, m
        center = i * m / n
        return max(0, math.floor(center) - width), min(m, math.ceil(center) + width)
    states = {(0, 0): (0.0, None, None)}
    transitions = [(1, 1)]
    if max_group >= 2:
        transitions += [(1, 2), (2, 1), (2, 2)]
    if max_group >= 3:
        transitions += [(1, 3), (3, 1)]
    def relax(i, j, ni, nj, extra, operation):
        if ni > n or nj > m:
            return
        lower, upper = bounds(ni)
        if not lower <= nj <= upper:
            return
        cost = states[i, j][0] + extra
        if (ni, nj) not in states or cost < states[ni, nj][0] - 1e-10:
            states[ni, nj] = cost, (i, j), operation
    for i in range(n + 1):
        lower, upper = bounds(i)
        for j in range(lower, upper + 1):
            if (i, j) not in states:
                continue
            if i < n:
                relax(i, j, i + 1, j, 0.46, (1, 0, None))
            if j < m:
                relax(i, j, i, j + 1, 0.46, (0, 1, None))
            for a, b in transitions:
                if i + a > n or j + b > m:
                    continue
                features = _candidate_features(en_groups, vi_groups, es + i, vs + j,
                                               a, b, ratio, threshold)
                if features is None:
                    continue
                _, confidence, _ = features
                cost = (1.0 - confidence) * (a + b) / 2 + 0.055 * (a + b - 2)
                relax(i, j, i + a, j + b, cost, (a, b, features))
    diagnostics = {"en_units": n, "vi_units": m, "banded": banded,
                   "band_width": width if banded else None, "states": len(states)}
    if (n, m) not in states:
        diagnostics["status"] = "no_path_all_unmatched"
        return [], diagnostics
    result, cursor = [], (n, m)
    while cursor != (0, 0):
        _, previous, operation = states[cursor]
        a, b, features = operation
        if a and b:
            result.append((es + previous[0], vs + previous[1], a, b, features))
        cursor = previous
    return list(reversed(result)), diagnostics


def align_document(en_text, vi_text, embedder, *, max_group=3, threshold=0.62,
                   max_unit_chars=900, max_dp_cells=120000):
    """Align one document pair, retaining exact strings/offsets and a full audit.

    ``coverage_en`` and ``coverage_vi`` are accepted non-whitespace characters
    divided by all non-whitespace characters in each ORIGINAL document. They
    are coverage measures, not alignment accuracy. The threshold is fixed by
    the caller; it is never lowered automatically to achieve a desired rate.
    """
    en_text, vi_text = str(en_text), str(vi_text)
    if max_group not in (1, 2, 3) or not 0 < threshold <= 1:
        raise ValueError("max_group must be 1..3 and threshold must be in (0,1]")
    if max_dp_cells < 1000:
        raise ValueError("max_dp_cells must be >= 1000")
    en_units = legal_atomic_spans(en_text, "en", max_unit_chars)
    vi_units = legal_atomic_spans(vi_text, "vi", max_unit_chars)
    totals = {"en": _nonspace_len(en_text), "vi": _nonspace_len(vi_text)}
    en_groups, vi_groups = _unit_groups(en_units, max_group), _unit_groups(vi_units, max_group)
    pairs, regions, semantic_anchor_count, heading_anchors = [], [], 0, []
    if totals["en"] and totals["vi"]:
        # Reuse embeddings of duplicate strings; callable can cache across docs.
        # Encode atomic legal spans once. Compose 2/3-span candidates from
        # length-weighted normalized atom vectors; this avoids ~3x LaBSE calls.
        unique = list(dict.fromkeys(groups[i, 1]["text"] for groups, units in
            ((en_groups, en_units), (vi_groups, vi_units)) for i in range(len(units))
            if (i, 1) in groups))
        vectors = _embedding_matrix(embedder, unique)
        lookup = dict(zip(unique, vectors))
        for groups, units in ((en_groups, en_units), (vi_groups, vi_units)):
            for i in range(len(units)):
                if (i, 1) in groups:
                    groups[i, 1]["embedding"] = lookup[groups[i, 1]["text"]]
            for (start, size), group in groups.items():
                if size == 1: continue
                members = [groups[k, 1] for k in range(start, start + size)]
                weights = np.array([max(1, x["chars"]) for x in members], dtype=np.float32)
                vector = np.average(np.stack([x["embedding"] for x in members]),
                                    axis=0, weights=weights)
                group["embedding"] = vector / max(np.linalg.norm(vector), 1e-12)
        ratio = min(2.5, max(0.4, totals["vi"] / totals["en"]))
        heading_anchors = _heading_anchors(en_units, vi_units, en_groups, vi_groups, threshold)
        coarse = sorted(set([(0, 0), *heading_anchors, (len(en_units), len(vi_units))]))
        boundaries = [coarse[0]]
        for (es, vs), (ee, ve) in zip(coarse, coarse[1:]):
            if (ee - es + 1) * (ve - vs + 1) > max_dp_cells:
                anchors = _semantic_anchors(en_groups, vi_groups, es, ee, vs, ve, threshold)
                anchors = [pair for pair in anchors if es < pair[0] < ee and vs < pair[1] < ve]
                semantic_anchor_count += len(anchors)
                boundaries.extend(anchors)
            boundaries.append((ee, ve))
        boundaries = sorted(set(boundaries))
        for (es, vs), (ee, ve) in zip(boundaries, boundaries[1:]):
            operations, diagnostics = _region_dp(en_groups, vi_groups, es, ee, vs, ve,
                                                  ratio, threshold, max_group, max_dp_cells)
            diagnostics.update(en_start_unit=es, vi_start_unit=vs)
            regions.append(diagnostics)
            for i, j, a, b, (cosine, confidence, numeric_penalty) in operations:
                en_start, en_end = en_units[i]["start"], en_units[i + a - 1]["end"]
                vi_start, vi_end = vi_units[j]["start"], vi_units[j + b - 1]["end"]
                pairs.append({"en": en_text[en_start:en_end], "vi": vi_text[vi_start:vi_end],
                              "en_start": en_start, "en_end": en_end,
                              "vi_start": vi_start, "vi_end": vi_end,
                              "score": cosine, "confidence": confidence,
                              "numeric_penalty": numeric_penalty,
                              "method": f"semantic_dp_{a}:{b}",
                              "en_unit_start": i, "en_unit_end": i + a,
                              "vi_unit_start": j, "vi_unit_end": j + b})
    unmatched = {}
    accepted_chars = {}
    for lang, units, original in (("en", en_units, en_text), ("vi", vi_units, vi_text)):
        accepted = set()
        for pair in pairs:
            accepted.update(range(pair[f"{lang}_unit_start"], pair[f"{lang}_unit_end"]))
        unmatched[lang] = [{"start": unit["start"], "end": unit["end"],
                            "text": unit["text"], "chars": unit["chars"],
                            "reason": "no_confident_monotone_match"}
                           for i, unit in enumerate(units) if i not in accepted]
        accepted_chars[lang] = sum(_nonspace_len(pair[lang]) for pair in pairs)
        assert accepted_chars[lang] + sum(unit["chars"] for unit in unmatched[lang]) == totals[lang]
        assert all(pair[lang] == original[pair[f"{lang}_start"]:pair[f"{lang}_end"]] for pair in pairs)
    assert all(a["en_end"] <= b["en_start"] and a["vi_end"] <= b["vi_start"]
               for a, b in zip(pairs, pairs[1:]))
    audit = {"version": ALIGNMENT_VERSION, "threshold": threshold,
             "total_chars_en": totals["en"], "total_chars_vi": totals["vi"],
             "accepted_chars_en": accepted_chars["en"], "accepted_chars_vi": accepted_chars["vi"],
             "units_en": len(en_units), "units_vi": len(vi_units), "accepted_pairs": len(pairs),
             "heading_anchors": len(heading_anchors), "semantic_anchors": semantic_anchor_count,
             "regions": regions, "unmatched_en": unmatched["en"], "unmatched_vi": unmatched["vi"],
             "coverage_is_not_precision": True}
    return {"pairs": pairs, "audit": audit,
            "coverage_en": accepted_chars["en"] / totals["en"] if totals["en"] else 0.0,
            "coverage_vi": accepted_chars["vi"] / totals["vi"] if totals["vi"] else 0.0}


