"""Exact source partition and conservative generation-loop detection.

No reference text is consulted. Retries regenerate source spans rather than
deleting generated words, and unresolved failures are never silently exported.
"""
import re

SAFETY_VERSION='exact_source_retry_v2'

def has_letters(text):
    return any(c.isalpha() for c in text)

def split_source(text,count_tokens,cap=512):
    parts=[];pos=0
    while pos<len(text):
        lo=pos+1;hi=len(text);best=pos
        while lo<=hi:
            mid=(lo+hi)//2
            if count_tokens(text[pos:mid])<=cap:best=mid;lo=mid+1
            else:hi=mid-1
        if best<=pos:raise RuntimeError('One character exceeds source token cap')
        boundary='end'
        if best<len(text):
            window=text[pos:best];minimum=len(window)//2
            marks=[m.end() for m in re.finditer(r'\n+|(?<=[.;!?])\s+',window) if m.end()>minimum]
            boundary='sentence_or_line'
            if not marks:
                marks=[m.end() for m in re.finditer(r'\s+',window) if m.end()>minimum]
                boundary='word'
            if marks:best=pos+marks[-1]
            else:boundary='character_fallback'
        raw=text[pos:best]
        assert count_tokens(raw)<=cap
        parts.append(dict(raw=raw,start=pos,end=best,boundary=boundary));pos=best
    assert ''.join(p['raw'] for p in parts)==text
    return parts

def split_retry(text):
    if len(text)<2:raise RuntimeError('Cannot split source further')
    mid=len(text)//2;lo=max(1,len(text)//4);hi=min(len(text)-1,3*len(text)//4)
    choices=[m.end() for m in re.finditer(r'\n+|(?<=[.;!?])\s+',text) if lo<=m.end()<=hi]
    kind='sentence_or_line'
    if not choices:
        choices=[m.end() for m in re.finditer(r'\s+',text) if lo<=m.end()<=hi];kind='word'
    if not choices:raise RuntimeError('No safe boundary to retry; manual review required')
    cut=min(choices,key=lambda x:abs(x-mid));parts=[text[:cut],text[cut:]]
    assert ''.join(parts)==text and all(parts)
    return parts,kind

def repetition_reason(source,text):
    if has_letters(source) and not text.strip():return 'empty'
    words=re.findall(r'\w+|[^\w\s]',text.casefold())
    source_words=re.findall(r'\w+|[^\w\s]',source.casefold())
    # Four columns often repeat translated labels. Require eight copies;
    # output-limit checks independently catch persistent generation loops.
    copies=8
    for width in range(1,9):
        for i in range(max(0,len(words)-copies*width+1)):
            phrase=words[i:i+width]
            if not any(has_letters(w) for w in phrase):continue
            if all(words[i+j*width:i+(j+1)*width]==phrase for j in range(1,copies)):
                if not any(source_words[k:k+copies*width]==phrase*copies for k in range(max(0,len(source_words)-copies*width+1))):
                    return 'consecutive_phrase_loop'
    # Pure symbol runs are only suspicious if expanded beyond their source run.
    for char in ('.','_','='):
        run=lambda s:max([len(m.group()) for m in re.finditer(re.escape(char)+r'{12,}',s)] or [0])
        if run(text)>max(64,2*run(source)+16):return 'expanded_symbol_loop'
    if has_letters(source) and not has_letters(text):return 'lost_all_text'
    return None

def literal_parts(source):
    """Dotted form fields are source structure, not language to generate."""
    return [p for p in re.split(r'([.\u2026_=\-]{12,})',source) if p]

def outer(text,translation):
    return re.match(r'^\s*',text).group()+translation.strip()+re.search(r'\s*$',text).group()

def repair_translation(source,generate,emit,depth=0,initial=None,fallback=None):
    if not has_letters(source):return source
    literals=literal_parts(source)
    if len(literals)>1:
        return ''.join(repair_translation(p,generate,emit,depth,fallback=fallback) for p in literals)
    text,hit_limit=initial if initial is not None else generate(source)
    reason='output_limit' if hit_limit else repetition_reason(source,text)
    if not reason:return outer(source,text)
    emit(dict(kind='retry',depth=depth,reason=reason,source_characters=len(source)))
    if fallback is not None:
        alternative,hit=fallback(source)
        alternative_reason='output_limit' if hit else repetition_reason(source,alternative)
        emit(dict(kind='fallback_decode',depth=depth,accepted=alternative_reason is None,reason=alternative_reason))
        if alternative_reason is None:return outer(source,alternative)
    if depth>=5:raise RuntimeError(f'Unresolved translation after retries: {reason}')
    pieces,boundary=split_retry(source)
    emit(dict(kind='retry_split',boundary=boundary,lengths=list(map(len,pieces))))
    return ''.join(repair_translation(p,generate,emit,depth+1,fallback=fallback) for p in pieces)
