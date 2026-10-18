"""Whitespace-only transfer from a same-language Qwen layout reference.

Never translates, fills omissions, deduplicates or substitutes reference words.
Token anchors transfer physical wrapping; unaligned intervals are interpolated
and explicitly reported, NOT certified as exact source semantic alignment.
"""
import bisect
import csv
import hashlib
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(r'E:\NCKH\Document-Level Legal')
REPORT = Path(__file__).parent / 'reports' / 'qwen_layout_transfer'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def tokenize(text):
    matches = list(re.finditer(r'\S+', text))
    words = [m.group() for m in matches]
    gaps = [text[:matches[0].start()]] if matches else [text]
    gaps += [text[a.end():b.start()] for a, b in zip(matches, matches[1:])]
    if matches:
        gaps.append(text[matches[-1].end():])
    return words, gaps


def normalize(word):
    return word.casefold().strip('.,;:!?()[]{}“”"\'') or word.casefold()


def unique_anchors(a, b, width=5):
    def positions(words):
        result = {}
        for i in range(len(words) - width + 1):
            key = tuple(words[i:i + width])
            result[key] = i if key not in result else -1
        return result
    ap, bp = positions(a), positions(b)
    pairs = sorted((i, bp[k]) for k, i in ap.items()
                   if i >= 0 and bp.get(k, -1) >= 0)
    tails, indices, previous = [], [], []
    for n, (_, j) in enumerate(pairs):
        slot = bisect.bisect_left(tails, j)
        previous.append(indices[slot - 1] if slot else -1)
        if slot == len(tails):
            tails.append(j); indices.append(n)
        else:
            tails[slot] = j; indices[slot] = n
    chain = []
    if indices:
        n = indices[-1]
        while n >= 0:
            chain.append(pairs[n]); n = previous[n]
    chain.reverse()
    accepted = []
    for i, j in chain:
        if not accepted or (i >= accepted[-1][0] + width and j >= accepted[-1][1] + width):
            accepted.append((i, j))
    return accepted


def boundary_map(reference, target):
    a, b = list(map(normalize, reference)), list(map(normalize, target))
    anchors = unique_anchors(a, b)
    blocks = []
    ai = bi = 0
    for i, j in anchors + [(len(a), len(b))]:
        if i > ai and j > bi:
            # Exact local matching is bounded to prevent quadratic global work.
            sm = SequenceMatcher(None, a[ai:i], b[bi:j], autojunk=max(i-ai, j-bi) > 600)
            blocks += [(ai+x.a, bi+x.b, x.size) for x in sm.get_matching_blocks() if x.size]
        if i < len(a):
            blocks.append((i, j, 5)); ai, bi = i+5, j+5
    knots = {0: 0, len(a): len(b)}
    exact = set()
    for i, j, size in blocks:
        for k in range(size + 1):
            # Prefer the right side of a target-only insertion at a boundary.
            knots[i+k] = max(knots.get(i+k, 0), j+k)
            exact.add(i+k)
    knots[0], knots[len(a)] = 0, len(b)
    ordered = sorted(knots.items())
    mapping = [0] * (len(a)+1)
    uncertain = []
    for (i, j), (end_i, end_j) in zip(ordered, ordered[1:]):
        if end_j < j:
            raise RuntimeError('Non-monotonic alignment')
        for x in range(i, end_i+1):
            mapping[x] = round(j+(end_j-j)*(x-i)/max(1, end_i-i))
        if end_i-i > 25 or end_j-j > 25:
            uncertain.append(dict(reference_word_start=i, reference_word_end=end_i,
                                  vinai_word_start=j, vinai_word_end=end_j))
    return mapping, exact, uncertain


def transfer(reference, original):
    rw, rg = tokenize(reference); tw, tg = tokenize(original)
    mapping, exact, uncertain = boundary_map(rw, tw)
    replacement = {}
    mapped_exact = mapped_interpolated = skipped_unaligned_columns = 0
    for i, gap in enumerate(rg):
        if '\n' not in gap and '\r' not in gap and '\t' not in gap and not re.search(r' {2,}', gap):
            continue
        j = mapping[i]
        if '\n' not in gap and '\r' not in gap:
            # Column gaps require an exact match on BOTH sides. Interpolating
            # a formula/table gap could split the wrong field or ordinary phrase.
            adjacent = (0 < i < len(rw) and 0 < j < len(tw)
                        and normalize(rw[i-1]) == normalize(tw[j-1])
                        and normalize(rw[i]) == normalize(tw[j]))
            if not adjacent:
                skipped_unaligned_columns += 1
                continue
        if j in (0, len(tw)) and i not in (0, len(rw)):
            continue
        if i in exact:
            mapped_exact += 1
        else:
            mapped_interpolated += 1
        if j in replacement:
            # A deleted reference span can map several breaks onto one boundary.
            # Keep the largest separation instead of inventing empty content rows.
            replacement[j] = max((replacement[j], gap), key=lambda s: (s.count('\n'), len(s)))
        else:
            replacement[j] = gap
    protected = 0
    for i in list(replacement):
        if 0 < i < len(tw):
            left, right = normalize(tw[i-1]), normalize(tw[i])
            phrase = (left, right) in {('nghị', 'định'), ('thông', 'tư'), ('quyết', 'định'),
                                       ('bộ', 'luật'), ('chapter', 'i'), ('chương', 'i'),
                                       ('thủy', 'văn')}
            marker = left in {'article', 'điều', 'chapter', 'chương', 'section', 'mục'} and bool(re.fullmatch(r'[0-9a-z]+', right))
            if '\n' in replacement[i] and (phrase or marker):
                del replacement[i]; protected += 1
    # Preamble paragraph starts are explicit existing target words. This also
    # prevents the title's final words being attached to the first legal basis
    # when Qwen accidentally left its reference preamble untranslated.
    structural_preamble_starts = 0
    for j in range(1, min(len(tw)-1, 1500)):
        if ((tw[j], normalize(tw[j+1])) == ('Căn', 'cứ')
                or tw[j] == 'Pursuant' and normalize(tw[j+1]) == 'to'):
            replacement[j] = '\n\n  '
            structural_preamble_starts += 1
    # Do not leave an English article/preposition dangling before a blank line
    # solely because translations have different word counts.
    paragraph_break_repairs = 0
    for j in list(replacement):
        gap = replacement.get(j, '')
        if 0 < j < len(tw) and gap.count('\n') >= 2 and normalize(tw[j-1]) in {'the','a','an','of','to','and','or','by','with','at','in','for'}:
            del replacement[j]
            if j >= 2 and re.search(r'[.;:!?]$', tw[j-2]):
                replacement[j-1] = gap
            paragraph_break_repairs += 1
    gaps = list(tg)
    for i, gap in enumerate(gaps):
        if i in replacement:
            gaps[i] = replacement[i]
        elif '\n' in gap or '\r' in gap:
            # The old model chunk boundary is not an intra-word space.
            gaps[i] = ' ' if 0 < i < len(tw) else gap
    # Target-only text (including original repeated material) has no reference
    # wrapping. Keep it all, but avoid giant lines; insert breaks at word gaps.
    wrapped_target_only_lines = 0
    length = len(gaps[0].rsplit('\n', 1)[-1])
    indent = re.match(r'[ \t]*', gaps[0].rsplit('\n', 1)[-1]).group()
    for i, word in enumerate(tw):
        length += len(word)
        gap = gaps[i+1]
        if '\n' in gap or '\r' in gap:
            tail = re.split(r'[\r\n]', gap)[-1]
            length = len(tail); indent = tail
        elif i+1 < len(tw) and length+len(gap)+len(tw[i+1]) > 150 and len(gap) <= 2:
            left, right = normalize(word), normalize(tw[i+1])
            if (left, right) not in {('nghị','định'), ('quyết','định'), ('thông','tư'), ('thủy','văn')} and left not in {'article','điều'}:
                gaps[i+1] = '\n'+indent
                length = len(indent); wrapped_target_only_lines += 1
            else:
                length += len(gap)
        else:
            length += len(gap)
    result = gaps[0] + ''.join(word+gaps[i+1] for i, word in enumerate(tw))
    after_words, after_gaps = tokenize(result)
    assert after_words == tw, 'Words/punctuation/order changed'
    assert re.sub(r'\s+', '', result) == re.sub(r'\s+', '', original)
    intra_line_spacing_preserved = all(
        before == after for before, after in zip(tg[1:-1], after_gaps[1:-1])
        if not re.search(r'[\r\n\t]', before+after) and '  ' not in before+after)
    assert intra_line_spacing_preserved
    audit = dict(reference_newlines=reference.count('\n'), output_newlines=result.count('\n'),
                 original_newlines=original.count('\n'), matched_layout_boundaries=mapped_exact,
                 interpolated_layout_boundaries=mapped_interpolated,
                 skipped_unaligned_column_gaps=skipped_unaligned_columns,
                 preamble_paragraph_starts=structural_preamble_starts,
                 repaired_dangling_english_paragraph_breaks=paragraph_break_repairs,
                 additional_long_line_wraps=wrapped_target_only_lines,
                 protected_heading_word_pairs=protected, unaligned_intervals=uncertain,
                 word_tokens_identical=True, nonwhitespace_sha256=hashlib.sha256(re.sub(r'\s+', '', original).encode()).hexdigest(),
                 intra_line_word_spacing_preserved=True, exact_source_layout_certified=False)
    return result, audit


def main():
    csv.field_size_limit(10000000)
    vinai_path = ROOT/'Output/VinAI_Private_20261015/results.csv'
    qwen_path = ROOT/'Output/private_test_submission/results.csv'
    inp = read(ROOT/'Dataset/private test/private_test_input.csv')
    reference = {(r['sample_id'], r['direction']): r['translation'] for r in read(qwen_path)}
    original = {(r['sample_id'], r['direction']): r['translation'] for r in read(vinai_path)}
    assert len(reference) == len(original) == 100 and reference.keys() == original.keys()
    rows, audits = [], []
    REPORT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for direction in ('en-vi', 'vi-en'):
        for doc in inp:
            if doc['source_language'] != ('en' if direction == 'en-vi' else 'vi'):
                continue
            key = (doc['sample_id'], direction)
            text, audit = transfer(reference[key], original[key])
            audit.update(sample_id=key[0], direction=direction, source_newlines=doc['source_markdown'].count('\n'))
            rows.append(dict(sample_id=key[0], direction=direction, translation=text))
            audits.append(audit)
            print(f'{len(rows)}/100 {key}: breaks={audit["output_newlines"]} unaligned={len(audit["unaligned_intervals"])}', flush=True)
    summary = dict(rows=100, content_unchanged_documents=100,
                   row_order='50 EN->VI then 50 VI->EN', source_layout_exact=False,
                   source_vinai_sha256=hashlib.sha256(vinai_path.read_bytes()).hexdigest(),
                   reference_qwen_sha256=hashlib.sha256(qwen_path.read_bytes()).hexdigest(),
                   newlines=sum(r['output_newlines'] for r in audits),
                   interpolated_boundaries=sum(r['interpolated_layout_boundaries'] for r in audits),
                   method='Same-target-language exact phrase anchors and monotonic word-boundary layout transfer; interpolate unmatched intervals',
                   limitation='Reference is a different translation, so source line/cell semantic correspondence is NOT certified. Existing errors/omissions/repetitions are retained.',
                   seconds=time.monotonic()-start, documents=audits)
    (REPORT/'rows.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf8')
    (REPORT/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    for sid in ['Trach_nhiem_hinh_su/43', 'Bat_dong_san/102', 'Quyen_dan_su/240']:
        row = next(r for r in rows if r['sample_id'] == sid)
        (REPORT/(sid.replace('/', '_')+'_preview.txt')).write_text(row['translation'], encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'documents'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
