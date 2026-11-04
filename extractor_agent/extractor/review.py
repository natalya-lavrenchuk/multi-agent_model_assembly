"""review.py -- one page per paper to check and correct what the pipeline extracted.
Works at every stage: right after sup.py (boundary to approve; the rest says what to run next) and
after symbols.py, laws.py and supmeta.py (everything).

    python -m extractor.review papers/kontoravdi_2010     # writes papers/<id>/review.html; open it in a browser

The page shows the Sup (description, model assumptions, which equations are in the model and why the
others are not), every law (equation rendered, description, law type, assumptions, roles, possible
misprint, what needs review) and every symbol (description, unit, value, Parameter/Variable, which
laws compute and use it). Descriptions, units, classes, law types, the Sup fields and the model's
equation list can be edited in the page;
'Export corrections' downloads corrections.json. Put that file in the paper's folder: graph.py
applies it on top of symbols.json and laws.json, so corrections survive a rerun of the LLM steps.
Loading an existing corrections.json is done by this script (it is built into the page).
"""
from __future__ import annotations
import html
import json
import os
import sys

from extractor import triage
from extractor.graph import _computes, _flag_misreads, _misreads
from extractor.paper import clean, load_model, models


def build(paper_dir: str, app: bool = False) -> str:
    """app=True: the page is served by app.py, which saves corrections and runs the next steps."""
    cfg = json.load(open(f'{paper_dir}/paper.json')) if os.path.exists(f'{paper_dir}/paper.json') else \
        {'sup': os.path.basename(os.path.normpath(paper_dir)), 'status': 'none'}  # before sup.py: equations only
    formulas, reads, _ = load_model(paper_dir, allow_proposed=True)
    eqs = [str(f.get('eq') or i + 1) for i, f in enumerate(formulas)]
    load = lambda name, key: json.load(open(f'{paper_dir}/{name}'))[key] if os.path.exists(f'{paper_dir}/{name}') else {}
    syms, laws = load('symbols.json', 'symbols'), load('laws.json', 'laws')
    meta = load('sup_meta.json', 'sups')
    fixes = load('fixes.json', 'fixes')
    _flag_misreads(paper_dir, laws)
    sups = models(cfg)
    corr = json.load(open(f'{paper_dir}/corrections.json')) if os.path.exists(f'{paper_dir}/corrections.json') else {}
    data = {
        'paper': {**{k: cfg.get(k) for k in ('sup', 'title', 'doi', 'excluded', 'checks', 'status')},
                  'folder': os.path.basename(os.path.normpath(paper_dir))},   # the app's address for this paper
        'sups': [{'id': m['id'], 'name': m.get('name'), 'equations': [str(e) for e in m.get('equations') or eqs],
                  'equations_quote': m.get('equations_quote'), 'boundary_basis': m.get('boundary_basis'),
                  'complete_model': m.get('complete_model'), 'model_assumptions': m.get('model_assumptions', []),
                  'meta': _with_facts(meta.get(m['id'], {}))} for m in sups],
        'fixes': fixes,
        'misreads': _misreads(paper_dir),
        'laws': [{'eq': n, 'latex': clean(f['latex']), 'page': f.get('page', 0) + 1, 'lhs': r['lhs'], 'rhs': r['rhs'],
                  'form': r['form'], 'computes': c, 'condition': r['condition'],
                  **{k: v for k, v in laws.get(n, {}).items() if k != 'llm'}, **_tiers(laws.get(n, {}))}
                 for n, f, r, c in zip(eqs, formulas, reads, _computes(reads))],
        'symbols': [{'symbol': s, **{k: v for k, v in d.items() if k != 'llm'}, **_tiers(d)} for s, d in syms.items()],
        'corrections': corr,
    }
    return PAGE.replace('__APP__', 'true' if app else 'false') \
               .replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')) \
               .replace('__TITLE__', html.escape(cfg.get('sup', 'paper')))


def _with_facts(meta: dict) -> dict:
    """Old sup_meta.json files (fixed fields) shown as facts too."""
    from extractor.supmeta import facts_of
    return {**meta, 'facts': facts_of(meta)} if meta else meta


def _tiers(item: dict) -> dict:
    """Review by exception: 'check' = what a person must look at; 'fyi' = inferred, shown collapsed."""
    r = item.get('review') or []
    return {'check': [x for x in r if triage.tier(x) == 'check'], 'fyi': [x for x in r if triage.tier(x) == 'fyi']}


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review: __TITLE__</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
:root{--bg:#fff;--fg:#1d2a30;--mut:#66737a;--line:#d9e0e3;--card:#f6f8f9;--p:#a8501a;--v:#16706d;--c:#4b5ea8;--warn:#b7791f;--bad:#b42318;--ok:#2f7d32}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Arial,sans-serif}
header{position:sticky;top:0;background:#0e3440;color:#fff;padding:10px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:2}
header a{color:#cfe3ea;text-decoration:none} header button{margin-left:auto}
main{max-width:1150px;margin:0 auto;padding:16px 20px 80px}
h3{font-size:17px} h2{border-bottom:2px solid var(--line);padding-bottom:4px;margin-top:32px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin:10px 0}
.card.flag{border-left:4px solid var(--warn)} .card.done{border-left:4px solid var(--ok)}
.eqrow{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.num{font-weight:700;min-width:44px} .tex{overflow-x:auto;max-width:100%}
.badge{display:inline-block;font-size:12px;padding:1px 7px;border-radius:10px;border:1px solid var(--line);background:#fff;margin-right:4px}
.stated{color:var(--ok);border-color:var(--ok)} .inferred{color:var(--warn);border-color:var(--warn)} .unknown{color:var(--bad);border-color:var(--bad)}
.parameter{color:var(--p);border-color:var(--p)} .variable{color:var(--v);border-color:var(--v)} .computed{color:var(--c);border-color:var(--c)}
.blocking{color:#fff;background:var(--bad);border-color:var(--bad)} .warning{color:var(--warn);border-color:var(--warn)}
.mut{color:var(--mut);font-size:13px} .quote{font-style:italic;color:var(--mut);font-size:13px}
.review{color:var(--bad);font-size:13px;margin:4px 0 0 0;padding-left:18px}
.issue{background:#fdecea;border-radius:4px;padding:4px 8px;font-size:13px;margin-top:6px}
textarea,input,select{font:inherit;border:1px solid var(--line);border-radius:4px;padding:3px 6px;background:#fff}
textarea{width:100%;min-height:36px;box-sizing:border-box} .edited{background:#fff7d6}
table{border-collapse:collapse;width:100%} th,td{border-bottom:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}
th{position:sticky;top:46px;background:#eef3f5;font-size:13px}
.chips .badge{margin-right:6px;cursor:pointer;font-size:15px;padding:2px 8px} .grid{display:grid;grid-template-columns:140px 1fr;gap:4px 12px}
label.acc{font-size:13px;white-space:nowrap}
</style></head><body>
<header><b>Review: __TITLE__</b><a href="#sup">Sup</a><a href="#laws">Laws</a><a href="#symbols">Symbols</a>
<label><input type="checkbox" id="onlyflag" checked> only what needs checking</label>
<span id="count" class="mut" style="color:#cfe3ea"></span>
<button id="export">Export corrections</button></header>
<main><div id="pipe"></div><div id="sup"></div><h2 id="laws">Laws</h2><div id="lawlist"></div>
<h2 id="symbols">Symbols</h2><table><thead><tr><th>Symbol</th><th>Class</th><th>Description</th><th>Unit</th><th>Value</th><th>Laws</th><th>Check</th></tr></thead><tbody id="symtab"></tbody></table></main>
<script>
const APP = __APP__;
window.addEventListener('error', e => { const b = document.createElement('div'); b.className = 'issue'; b.style.margin = '8px 0';   // never fail silently
  b.textContent = 'The page hit an error while drawing (' + e.message + '). The step buttons still work; please report this.'; document.querySelector('main').prepend(b); });
const D = __DATA__;
const C = JSON.parse(JSON.stringify(D.corrections || {}));  C.symbols = C.symbols || {}; C.laws = C.laws || {}; C.sup = C.sup || {}; C.new_models = C.new_models || [];
const esc = s => (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const MACROS = Object.fromEntries(['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'eta', 'theta', 'kappa', 'lambda', 'mu', 'nu', 'pi', 'rho', 'sigma', 'tau', 'phi', 'chi', 'psi', 'omega'].map(g => ['\\up' + g, '\\' + g]));   // \upmu (upright Greek) is not in KaTeX
const tex = (s, display) => { try { return katex.renderToString(s, {displayMode: !!display, throwOnError: false, strict: false, macros: {...MACROS}}); } catch(e) { return esc(s); } };
const texOrRaw = s => { try { return katex.renderToString(s, {displayMode: true, throwOnError: true, strict: false, macros: {...MACROS}}); }   // OCR text that is not valid LaTeX
  catch (e) { return `<div class="mut">not a readable equation; the OCR text was:</div><code style="font-size:12px;word-break:break-all">${esc(s)}</code>`; } };
const bare = x => x.split(/(\\?[A-Za-z]+(?:[_^]\{[^{}]*\})+)/g).map((y, i) => i % 2 ? tex(y) : esc(y)).join('');   // q_{A} written without $
const mtext = s => (s || '').split(/(\\\(.*?\\\)|\$[^$]+\$)/g).map((x, i) => i % 2 ? tex(x.replace(/^\\\(|\\\)$|^\$|\$$/g, '')) : bare(x)).join('');   // prose with inline math
const badge = (t, cls) => t ? `<span class="badge ${cls || t}">${esc(t)}</span>` : '';
const symInfo = Object.fromEntries(D.symbols.map(s => [s.symbol, s]));
const val = (kind, id, field, orig) => (C[kind][id] && field in C[kind][id]) ? C[kind][id][field] : orig;
function set(kind, id, field, v, orig, el) {
  C[kind][id] = C[kind][id] || {};
  if (v === orig) delete C[kind][id][field]; else C[kind][id][field] = v;
  if (!Object.keys(C[kind][id]).length) delete C[kind][id];
  if (el) el.classList.toggle('edited', v !== orig); count();
}
function count() { const n = Object.keys(C.symbols).length + Object.keys(C.laws).length + Object.keys(C.sup).length + C.new_models.length; document.getElementById('count').textContent = n ? `${n} items changed or accepted` : ''; }

// ---- Sup
const P = D.paper;
const ranges = list => { const n = list.map(String), out = []; let i = 0;
  while (i < n.length) { let j = i; while (j + 1 < n.length && /^\d+$/.test(n[j]) && +n[j + 1] === +n[j] + 1) j++;
    out.push(j > i + 1 ? `${n[i]}–${n[j]}` : j === i + 1 ? `${n[i]}, ${n[j]}` : n[i]); i = j + 1; } return out.join(', '); };
function supCard(m) {
  const id = m.id, M = m.meta || {}, ok = val('sup', id, 'approved', false);
  const facts = M.facts || [], factText = facts.map(f => `${f.topic}: ${f.text}`).join('\n');
  const byTopic = {}; facts.forEach(f => (byTopic[f.topic] = byTopic[f.topic] || []).push(f));
  const factList = Object.entries(byTopic).map(([t, fs]) => `<div style="margin:4px 0"><b class="mut">${esc(t)}</b>${fs.map(f =>
    `<div style="margin-left:12px">${mtext(f.text)} ${badge(f.basis)}${f.quote ? ` <span class="quote">"${mtext(f.quote)}"</span>` : ''}</div>`).join('')}</div>`).join('') || '-';
  const symList = list => (list || []).map(s => `<span class="badge ${(symInfo[s] || {}).class || 'variable'}" title="${esc((symInfo[s] || {}).description)}">${tex(s)}</span>`).join(' ') || '-';
  // every symbol of this model's laws, split only by class: one colour per class, as in the law cards
  const used = [...new Set(D.laws.filter(l => m.equations.map(String).includes(String(l.eq))).flatMap(l => [...l.lhs, ...l.rhs]))];
  const ofClass = c => used.filter(s => ((symInfo[s] || {}).class || 'variable') === c);
  return `<div class="card ${ok ? 'done' : ''}"><div style="display:flex;align-items:baseline;gap:10px"><h3 style="margin:0"><input style="font:inherit;font-weight:700;width:420px;max-width:100%" data-k="sup" data-id="${esc(id)}" data-f="name" data-o="${esc(m.name)}" value="${esc(val('sup', id, 'name', m.name))}" title="model name (editable)"></h3>
    <span class="mut">Sup id ${esc(id)}</span><span style="margin-left:auto">${P.status === 'proposed' ? '<b style="color:var(--warn)">boundary proposed: edit, then Approve boundary</b>' : P.status === 'none' ? '' : '<span style="color:var(--ok)">boundary approved</span>'}</span></div>
  <div class="grid" style="margin-top:8px">
  <b>Model equations</b><span><input style="width:100%;box-sizing:border-box" data-k="sup" data-id="${esc(id)}" data-f="equations" data-o="${esc(ranges(m.equations))}" value="${esc(val('sup', id, 'equations', ranges(m.equations)))}">
    ${D.sups.length > 1 && P.status === 'proposed' ? '<span class="mut">empty the list to drop this model</span> ' : ''}${badge(m.boundary_basis || '')} ${m.equations_quote ? `<span class="quote">"${mtext(m.equations_quote)}"</span>` : ''}</span>
  ${M.description === undefined ? '<b>Description</b><span class="review">not written yet: run supmeta.py (after symbols.py and laws.py)</span>' :
   `
  <b>Description</b><span><textarea style="min-height:70px" data-k="sup" data-id="${esc(id)}" data-f="description" data-o="${esc(M.description)}">${esc(val('sup', id, 'description', M.description))}</textarea></span>
  <b>What the paper says</b><span><span class="mut">${facts.length} facts, ${Object.keys(byTopic).length} topics (topics chosen by the LLM)</span>${factList}
    <details><summary class="mut">edit facts</summary><textarea style="min-height:160px;width:100%" data-k="sup" data-id="${esc(id)}" data-f="facts" data-o="${esc(factText)}">${esc(val('sup', id, 'facts', factText))}</textarea><span class="mut">one per line, topic: text. Unchanged lines keep their quote.</span></details></span>
  <b>Variables (${ofClass('variable').length})</b><span>${symList(ofClass('variable'))}</span>
  <b>Parameters (${ofClass('parameter').length})</b><span>${symList(ofClass('parameter'))}</span>`}
  <b>Complete model</b><span>${m.complete_model === undefined || m.complete_model === null ? '-' : m.complete_model}</span>
  <b>Model assumptions</b><span>${(m.model_assumptions || []).map(a => `<div>${badge(a.severity, a.severity)}${a.eq ? `<span class="badge">eq ${esc(a.eq)} only</span>` : ''} ${mtext(a.text)}<div class="quote">"${mtext(a.quote)}"</div></div>`).join('') || '-'}</span>
  </div></div>`;
}
document.getElementById('sup').innerHTML = `<h2>Sup${D.sups.length > 1 ? 's' : ''}: ${mtext(P.title)}</h2>
  <div class="mut">DOI ${esc(P.doi)} · ${D.sups.length} model${D.sups.length > 1 ? 's (rival models: one Sup each)' : ''}</div>
  ${P.status === 'none' ? `<div class="issue" style="margin:8px 0">No model boundary yet: ${APP ? 'press <b>Propose boundary</b> above' : 'run python -m extractor.sup'}. Below are all the printed equations as read.</div>` : ''}
  ${P.status === 'proposed' ? `<div class="issue" style="margin:8px 0">The model boundary is a proposal. Check the model equations below (edit the list if needed), then ${APP ? 'press <b>Approve boundary</b> above (it does not start the extraction)' : `Export corrections, put corrections.json in the paper folder and run: python -m extractor.sup ${esc('papers/' + P.folder)} --approve`}.</div>` : ''}
  ${D.sups.map(supCard).join('')}
  <div id="newmodels"></div>
  <button id="addmodel" style="margin:6px 0 10px">+ Add a model (another Sup from this paper)</button>
  <div class="card"><b>Left out of ${D.sups.length > 1 ? 'all models' : 'the model'}</b>
  ${(P.excluded || []).map(e => `<div>eq ${esc(e.eq)}: <span class="mut">${mtext(e.reason)}</span></div>`).join('') || '<div class="mut">nothing</div>'}
  ${(P.checks || []).length ? `<ul class="review">${P.checks.map(c => `<li>${esc(c)}</li>`).join('')}</ul>` : ''}</div>`;

// ---- possible misprint: the PDF next to the equation the model will use
const STRUCT = new Set(['\\frac', '\\dfrac', '\\left', '\\right', '\\mathrm', '\\mathbf', '\\boldsymbol', '\\begin', '\\end', '\\sqrt', '\\text',
  '\\operatorname', '\\big', '\\Big', '\\bigg', '\\Bigg', '\\displaystyle', '\\limits', '\\mathit', '\\hat', '\\bar', '\\dot', '\\tilde', '\\overline']);
const TOK = /\\[a-zA-Z]+|\\.|[a-zA-Z0-9]|\S/g;
function highlight(base, next) {           // colour what `next` has that `base` lacks (diff over LaTeX tokens)
  const a = base.match(TOK) || [], b = [...next.matchAll(TOK)], n = a.length, m = b.length;
  if (!n || !m || n * m > 400000) return tex(next, true);
  const T = Array.from({length: n + 1}, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    T[i][j] = a[i] === b[j][0] ? T[i + 1][j + 1] + 1 : Math.max(T[i + 1][j], T[i][j + 1]);
  const keep = new Set(), near = new Set(); let i = 0, j = 0;
  while (i < n && j < m) { if (a[i] === b[j][0]) { keep.add(j); i++; j++; } else if (T[i + 1][j] >= T[i][j + 1]) { near.add(j); near.add(j - 1); i++; } else j++; }
  if (i < n) near.add(m - 1);                                // something was removed: mark the atoms beside the gap
  const atom = t => /^[A-Za-z0-9+\-=\/]$/.test(t) || (/^\\[a-zA-Z]+$/.test(t) && !STRUCT.has(t));
  [...near].forEach(k => { let q = Math.min(k, m - 1); while (q >= 0 && q > k - 4 && !atom(b[q][0])) q--; if (q >= 0 && atom(b[q][0])) keep.delete(q); });   // nearest atom before the gap
  let out = '', last = 0;
  b.forEach((x, k) => { if (!keep.has(k) && atom(x[0])) { out += next.slice(last, x.index) + `{\\color{#c2410c}{${x[0]}}}`; last = x.index + x[0].length; } });
  out += next.slice(last);
  try { return katex.renderToString(out, {displayMode: true, throwOnError: true, strict: false, macros: {...MACROS}}); } catch (e) { return tex(next, true); }
}
const clean = s => (s || '').replace(/\$\$|\\\[|\\\]/g, '').trim();
const defChoice = id => (D.misreads || {})[id] ? 'ocr' : ((D.fixes || {})[id] || {}).action === 'correct' ? 'fix' : 'paper';
function shownLatex(L) {                     // the equation as it stands: an unreviewed OCR misread shows what the OCR read
  const M = (D.misreads || {})[String(L.eq)];
  return M && val('laws', String(L.eq), 'choice', 'ocr') === 'ocr' ? clean(M.printed_latex) : L.latex; }
function misprintCard(L) {
  const id = String(L.eq), F = (D.fixes || {})[id] || {}, M = (D.misreads || {})[id];
  const read = M ? clean(M.printed_latex) : clean(L.latex), paperV = M ? clean(M.corrected_latex) : clean(L.latex);
  const proposed = F.action === 'correct' && F.corrected_latex ? clean(F.corrected_latex) : null;
  const def = defChoice(id), choice = val('laws', id, 'choice', def);
  const own = val('laws', id, 'own_latex', proposed || paperV);
  const prints = val('laws', id, 'paper_prints_it', ''), already = val('laws', id, 'already_in_paper', false);
  const use = choice === 'fix' && proposed ? proposed : choice === 'own' ? clean(own) : choice === 'ocr' && M ? read : paperV;
  const outcome = choice === 'ocr' && M ? 'not reviewed yet / kept as the OCR read it'
    : choice === 'own' && !prints && use !== paperV ? 'answer the question above'
    : (choice === 'fix' && already) || (choice === 'own' && prints === 'yes') ? 'our reading was wrong: the stored equation is replaced to match the PDF'
    : choice === 'fix' || choice === 'own' ? (M ? 'our reading was wrong, and ' : '') + 'error in the paper: the printed equation is kept, the model uses the corrected one'
    : M ? 'our reading was wrong: the stored equation is replaced to match the PDF' : 'false alarm: the model uses the equation as printed';
  const why = choice === 'fix' && proposed ? `${mtext(F.reason)}${F.quote ? `<div class="quote">"${mtext(F.quote.replace(/^["“”\s]+|["“”\s]+$/g, ''))}"</div>` : ''}${(F.checks || []).length ? `<div class="review">check: ${F.checks.map(esc).join('; ')}</div>` : ''}`
    : choice === 'paper' && M ? esc(M.note || '') : choice === 'own' ? 'your version' : '';
  const opt = (v, t) => `<label style="margin-right:14px"><input type="radio" name="ch-${esc(id)}" data-k="laws" data-id="${esc(id)}" data-f="choice" data-o="${def}" value="${v}" ${choice === v ? 'checked' : ''}> ${t}</label>`;
  const row = (label, body) => `<div style="display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;margin:6px 0"><b class="mut">${label}</b><div>${body}</div></div>`;
  const flagged = L.issue ? mtext(L.issue) : M ? esc(M.note || '') : '';
  const opts = (proposed ? opt('fix', '<b>Accept the change</b>') : '')
    + opt('paper', M ? (proposed ? "Keep the paper's version (read correctly)" : '<b>Use the corrected reading</b>') : proposed ? "Keep the paper's version" : '<b>Use this</b>')
    + (M ? opt('ocr', 'Keep what the OCR read') : '') + opt('own', 'Edit');
  if (!proposed && !M && choice !== 'own')          // nothing to compare: the flag, why no change, and the choice
    return `<div style="margin:6px 0;padding:8px 10px;border:1px solid var(--line);border-left:4px solid var(--warn);border-radius:6px;background:#fffdf7">
    ${row('Why it was flagged', flagged)}${F.reason ? row('No change proposed', mtext(F.reason)) : ''}
    <div style="margin-top:8px">${opts}</div><div class="mut" style="margin-top:6px">Recorded as: ${outcome}</div></div>`;
  return `<div style="margin:6px 0;padding:8px 10px;border:1px solid var(--line);border-left:4px solid var(--warn);border-radius:6px;background:#fffdf7">
    ${row('In the paper', APP ? `<img src="/crop/${encodeURIComponent(P.folder)}/${encodeURIComponent(id)}.png" alt="equation in the PDF" style="max-width:100%;border:1px solid #ddd;background:#fff" onerror="this.replaceWith(document.createTextNode('(no picture: this equation was read from a table)'))">` : '<span class="mut">picture shown in the app</span>')}
    ${M && use !== read ? row('The OCR read', texOrRaw(read)) : ''}
    ${row(choice === 'fix' ? 'Proposed change' : choice === 'own' ? 'Your version' : choice === 'ocr' && M ? 'As the OCR read it' : M ? 'Corrected reading' : 'Paper’s version, kept', `<div id="pv-${esc(id)}">${use === read ? (M ? texOrRaw(use) : tex(use, true)) : M && texOrRaw(read).includes('not a readable equation') ? tex(use, true) : highlight(read, use)}</div>`)}
    ${use !== read && why ? row('Reason', why) : ''}
    ${row('Why it was flagged', flagged)}
    <div style="margin-top:8px">${opts}</div>
    ${choice === 'fix' && proposed && !M ? `<label class="mut" style="display:block;margin-top:4px"><input type="checkbox" data-k="laws" data-id="${esc(id)}" data-f="already_in_paper" ${already ? 'checked' : ''}> tick only if the picture above already shows this change: then the paper is right and our OCR misread it</label>` : ''}
    ${choice === 'own' ? `<div style="margin-top:6px"><div class="mut">The equation the model should use (LaTeX):</div>
       <textarea style="width:100%;min-height:50px" data-k="laws" data-id="${esc(id)}" data-f="own_latex" data-o="${esc(proposed || paperV)}">${esc(own)}</textarea>
       <div style="margin-top:4px">Does the paper print it this way?
       ${[['yes', 'Yes, our reading was wrong'], ['no', 'No, the paper has it wrong']].map(([v, t]) => `<label style="margin-left:10px"><input type="radio" name="pp-${esc(id)}" data-k="laws" data-id="${esc(id)}" data-f="paper_prints_it" data-o="" value="${v}" ${prints === v ? 'checked' : ''}> ${t}</label>`).join('')}</div></div>` : ''}
    <div class="mut" style="margin-top:6px">Recorded as: ${outcome}</div></div>`;
}

// ---- Laws
function lawCard(L) {
  const id = L.eq, review = L.check || [], fyi = L.fyi || [], done = val('laws', id, 'accepted', false);
  const chips = [...new Set([...L.lhs, ...L.rhs])].map(s => {
    const i = symInfo[s] || {}; const cls = i.class || 'variable';      // variable or parameter; what the law computes is named above
    return `<span class="badge ${cls}" title="${esc(i.description)}" onclick="location.hash='s-'+encodeURIComponent('${esc(s).replace(/\\/g,'\\\\')}')">${tex(s)}</span>`; }).join('');
  return `<div class="card ${done ? 'done' : review.length ? 'flag' : ''}" data-flag="${review.length && !done ? 1 : 0}" id="l-${esc(id)}">
  <div class="eqrow"><span class="num">(${esc(id)})</span><span class="tex">${tex(shownLatex(L), true)}</span></div>
  ${D.sups.length > 1 ? `<div class="mut">in: ${D.sups.filter(m => m.equations.includes(String(id))).map(m => esc(m.name)).join(', ') || 'no model'}</div>` : ''}
  <div class="mut">p.${L.page} · ${esc(L.section || '')} · ${esc(L.form)} · computes ${L.computes ? tex(L.computes) : '<b style="color:var(--bad)">undecided</b>'}${L.condition ? ' · side condition ' + tex(L.condition) : ''}</div>
  <div class="chips" style="margin:6px 0">${chips}</div>
  ${L.description === undefined ? '<div class="review">no law metadata yet: run laws.py</div>' : `
  <div style="display:flex;gap:10px;align-items:center;margin:4px 0">
    <b>Description</b> ${badge(L.description_basis)}
    ${review.length ? `<label class="acc" style="margin-left:auto"><input type="checkbox" data-k="laws" data-id="${esc(id)}" data-f="accepted" ${done ? 'checked' : ''}> resolved</label>` : ''}</div>
  <textarea data-k="laws" data-id="${esc(id)}" data-f="description" data-o="${esc(L.description)}">${esc(val('laws', id, 'description', L.description))}</textarea>
  ${L.quote ? `<div class="quote">"${mtext(L.quote)}"</div>` : ''}
  ${(L.assumptions || []).map(a => `<div>${badge(a.severity, a.severity)} ${mtext(a.text)} ${a.quote ? `<span class="quote">"${mtext(a.quote)}"</span>` : '<span class="mut">(not quoted)</span>'}</div>`).join('')}
  ${(L.roles || []).length && !['ODE', 'PDE'].includes(L.form) ? `<div class="mut">roles: ${L.roles.map(r => `${esc(r.role)} = ${tex(r.symbol)}`).join(' · ')}</div>` : ''}
  ${L.issue || (D.misreads || {})[String(L.eq)] ? misprintCard(L) : ''}`}
  ${review.filter(r => !r.startsWith('possible misprint') && !r.startsWith('possible OCR misread')).length ? `<ul class="review">${review.filter(r => !r.startsWith('possible misprint') && !r.startsWith('possible OCR misread')).map(r => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
  ${fyi.length ? `<details class="mut"><summary>for information (${fyi.length})</summary><ul>${fyi.map(r => `<li>${esc(r)}</li>`).join('')}</ul></details>` : ''}</div>`;
}
document.getElementById('lawlist').innerHTML =
  `<div class="card"><b>To check: ${D.laws.filter(l => (l.check || []).length).length + D.symbols.filter(x => (x.check || []).length).length}</b>
   <span class="mut"> · for information only: ${D.laws.filter(l => !(l.check || []).length && (l.fyi || []).length).length + D.symbols.filter(x => !(x.check || []).length && (x.fyi || []).length).length} (inferred, not wrong; shown collapsed)</span></div>` + D.laws.map(lawCard).join('');

// ---- Symbols
function symRow(s) {
  const id = s.symbol, computed = (s.computed_by || []).length, review = s.check || [], fyi = s.fyi || [], done = val('symbols', id, 'accepted', false);
  const cls = val('symbols', id, 'class', s.class);
  return `<tr id="s-${encodeURIComponent(id)}" data-flag="${review.length && !done ? 1 : 0}">
  <td>${tex(id)}</td>
  <td>${computed ? badge('variable', 'variable') + '<div class="mut">computed by a law</div>' : `<select data-k="symbols" data-id="${esc(id)}" data-f="class" data-o="${esc(s.class)}" class="${cls}">${['parameter', 'variable'].map(c => `<option ${cls === c ? 'selected' : ''}>${c}</option>`).join('')}</select><div class="mut">${badge(s.class_basis)}</div>`}</td>
  <td><textarea data-k="symbols" data-id="${esc(id)}" data-f="description" data-o="${esc(s.description)}">${esc(val('symbols', id, 'description', s.description))}</textarea>
      ${badge(s.description_basis)} ${s.quote ? `<span class="quote">"${mtext(s.quote)}"</span>` : ''}
      ${!computed && s.class_reason ? `<div class="mut">class: ${mtext(s.class_reason)}</div>` : ''}
      ${[['same quantity as', s.same_as, null], ['uncertainty', s.uncertainty, null], ['estimation', s.estimation, s.estimation_quote], ['measured', s.measured, s.measured_quote], ['initial value', s.initial_value, s.initial_quote]]
        .filter(([, v]) => v && v !== 'not said').map(([k, v, q]) => `<div class="mut"><b>${k}:</b> ${mtext(v)}${q ? ` <span class="quote">"${mtext(q)}"</span>` : ''}</div>`).join('')}
      ${review.length ? `<ul class="review">${review.map(r => `<li>${esc(r)}</li>`).join('')}</ul>` : ''}
      ${fyi.length ? `<details class="mut"><summary>for information (${fyi.length})</summary><ul>${fyi.map(r => `<li>${esc(r)}</li>`).join('')}</ul></details>` : ''}</td>
  <td><input size="10" data-k="symbols" data-id="${esc(id)}" data-f="unit" data-o="${esc(s.unit)}" value="${esc(val('symbols', id, 'unit', s.unit))}"><div>${badge(s.unit_basis)}</div></td>
  <td class="mut">${s.value ? tex(s.value.replace(/×/g, '\\times ')) : ''}${s.table ? `<div title="${esc(s.table)}">table ✓</div>` : ''}</td>
  <td class="mut">${computed ? 'computed by ' + s.computed_by.join(', ') + '<br>' : ''}${(s.used_in || []).length ? 'used in ' + s.used_in.join(', ') : ''}</td>
  <td>${review.length ? `<label class="acc"><input type="checkbox" data-k="symbols" data-id="${esc(id)}" data-f="accepted" ${done ? 'checked' : ''}> resolved</label>` : ''}</td></tr>`;
}
document.getElementById('symtab').innerHTML = D.symbols.length ? `<tr id="symnone"><td colspan="7" class="mut">No symbol needs checking. Untick "only what needs checking" at the top to see all ${D.symbols.length}.</td></tr>` + D.symbols.map(symRow).join('') : '<tr><td colspan="7" class="review">no symbols.json yet: run symbols.py</td></tr>';

// ---- editing
document.addEventListener('change', e => {
  const t = e.target, k = t.dataset.k; if (!k) return;
  const MP = ['choice', 'paper_prints_it', 'already_in_paper'];
  if (t.type === 'checkbox') set(k, t.dataset.id, t.dataset.f, t.checked, false);
  else set(k, t.dataset.id, t.dataset.f, t.value, t.dataset.o, t);
  if (k === 'laws' && MP.includes(t.dataset.f)) {          // a choice on a possible misprint: it resolves the flag; redraw the card
    set('laws', t.dataset.id, 'accepted', true, false);
    const L = D.laws.find(l => String(l.eq) === t.dataset.id), el = document.getElementById('l-' + t.dataset.id);
    if (L && el) el.outerHTML = lawCard(L); return; }   // stays on screen until the next reload, like 'resolved'
  if (t.type === 'checkbox') { t.closest('.card,tr').classList.toggle('done', t.checked); return; }
  if (t.tagName === 'SELECT' && k === 'symbols') t.className = t.value;
});
document.addEventListener('input', e => { const t = e.target; if (t.dataset.k && t.tagName !== 'SELECT' && t.type !== 'checkbox' && t.type !== 'radio') set(t.dataset.k, t.dataset.id, t.dataset.f, t.value, t.dataset.o, t);
  if (t.dataset.f === 'own_latex') { const pv = document.getElementById('pv-' + t.dataset.id), L = D.laws.find(l => String(l.eq) === t.dataset.id);
    if (pv && L) { const M = (D.misreads || {})[t.dataset.id]; pv.innerHTML = highlight(clean(M ? M.printed_latex : L.latex), clean(t.value)); } } });
const filter = on => { document.querySelectorAll('[data-flag]').forEach(el => el.style.display = on && el.dataset.flag === '0' ? 'none' : '');
  const hidden = on && D.symbols.length && !document.querySelector('#symtab tr[data-flag="1"]');
  const none = document.getElementById('symnone'); if (none) none.style.display = hidden ? '' : 'none'; };   // no row before symbols.py has run
document.getElementById('onlyflag').addEventListener('change', e => filter(e.target.checked));
filter(true);                                              // open on what needs checking; untick to see everything
// ---- new models (rival Sups the LLM did not propose); applied by sup.py --approve
function drawNew() {
  const opts = D.sups.map(m => `<option value="${esc(m.id)}">${esc(m.name || m.id)}</option>`).join('');
  document.getElementById('newmodels').innerHTML = C.new_models.map((nm, i) => `<div class="card" style="border-left:4px solid var(--c)">
    <div class="grid"><b>New model</b><span><input style="width:100%;box-sizing:border-box" data-new="${i}" data-nf="name" value="${esc(nm.name)}" placeholder="name, e.g. Simplified kinetics, dynamic φ"></span>
    <b>Model equations</b><span><input style="width:100%;box-sizing:border-box" data-new="${i}" data-nf="equations" value="${esc(nm.equations)}" placeholder="e.g. C-11–C-16, 10, 24, 25, 28, 33"></span>
    <b>Based on</b><span><select data-new="${i}" data-nf="based_on">${opts.replace(`value="${esc(nm.based_on)}"`, `value="${esc(nm.based_on)}" selected`)}</select>
      <span class="mut">its model assumptions and "complete model" are copied</span>
      <button style="margin-left:12px" onclick="C.new_models.splice(${i},1);drawNew();count()">remove</button></span></div></div>`).join('');
}
document.getElementById('addmodel').addEventListener('click', () => { C.new_models.push({name: '', equations: '', based_on: D.sups[0].id}); drawNew(); count(); });
document.addEventListener('input', e => { const t = e.target; if (t.dataset.new !== undefined) C.new_models[+t.dataset.new][t.dataset.nf] = t.value; });
document.addEventListener('change', e => { const t = e.target; if (t.dataset.new !== undefined) C.new_models[+t.dataset.new][t.dataset.nf] = t.value; });
drawNew();

</script>
<script>   // the step panel and saving: its own script, so an error drawing the cards above never removes the buttons
const corrections = () => ({paper: P.sup, sup: C.sup, new_models: C.new_models.filter(m => m.equations.trim()), note: 'human corrections; graph.py applies them on top of symbols.json and laws.json', symbols: C.symbols, laws: C.laws});
async function save() {
  const r = await fetch(`/api/${P.folder}/corrections`, {method: 'POST', body: JSON.stringify(corrections())});
  const b = document.getElementById('export'); b.textContent = r.ok ? 'Saved' : 'Save failed'; setTimeout(() => b.textContent = 'Save corrections', 1500); return r.ok; }
if (APP) document.getElementById('export').textContent = 'Save corrections';
document.getElementById('export').addEventListener('click', () => {
  if (APP) return save();
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(corrections(), null, 1)], {type: 'application/json'}));
  a.download = 'corrections.json'; a.click(); });

// ---- pipeline panel (app.py only): where this paper stands, and the button for the next step
const STEPS = [['ocr', 'OCR'], ['boundary', 'Boundary'], ['symbols', 'Symbols'], ['laws', 'Laws'], ['supmeta', 'Sup description'], ['graph', 'Graph']];
async function run(job) { await save(); await fetch(`/api/${P.folder}/run`, {method: 'POST', body: JSON.stringify({job})}); poll(); }
async function poll() {
  const st = await (await fetch(`/api/${P.folder}/status`)).json();
  const mark = v => v === 'done' ? '<span style="color:var(--ok)">✓</span>' : v === 'waiting' ? '<span style="color:var(--warn)">●</span>' : v === 'stale' ? '<span style="color:var(--warn)" title="older than its inputs: rebuild">↻</span>' : '<span class="mut">○</span>';
  const b = (job, label, main) => `<button ${st.running ? 'disabled' : ''} onclick="run('${job}')" style="${main ? 'font-weight:700' : ''}">${label}</button>`;
  const S = st.stages, appr = S.boundary === 'done';
  const grp = (title, items) => `<div style="margin-top:8px"><span class="mut" style="display:inline-block;width:150px">${title}</span>${items.join(' ')}</div>`;
  const bb = (job, label, main, ok) => `<button ${st.running || ok === false ? 'disabled' : ''} onclick="run('${job}')" style="${main ? 'font-weight:700' : ''}">${label}</button>`;
  const boundary = S.boundary === 'none' ? [bb('sup', 'Propose boundary (LLM)', 1)]
    : S.boundary === 'waiting' ? [bb('approve_only', 'Approve boundary', 1), bb('approve', 'Approve and run extraction'), bb('sup', 'Propose again (LLM)')]
    : ['<span style="color:var(--ok)">approved</span>', bb('approve_only', 'Re-approve with my edits')];
  const extract = [bb('symbols', 'Symbols', 0, appr), bb('laws', 'Laws', 0, appr), bb('supmeta', 'Sup description', 0, appr),
    bb('extract', 'All three', [S.symbols, S.laws, S.supmeta].includes('none'), appr)];
  const buttons = grp('1. Boundary', boundary) + grp('2. Extraction (LLM)', extract)
    + grp('3. Graph (no LLM)', [bb('graph', 'Save edits and rebuild graph', S.graph !== 'done', appr)])
    + grp('Tools', [bb('estimate', 'Estimate tokens', 0, appr)]);
  document.getElementById('pipe').innerHTML = `<div class="card" style="margin-top:16px">
    <div style="display:flex;gap:18px;flex-wrap:wrap">${STEPS.map(([k, l]) => `<span>${mark(S[k])} ${l}</span>`).join('')}
    <span style="margin-left:auto">${S.pending ? `<b style="color:var(--bad)">${S.pending} to check</b>` : '<span style="color:var(--ok)">nothing to check</span>'}${S.resolved ? ` <span class="mut">· ${S.resolved} resolved</span>` : ''}${S.fyi ? ` <span class="mut">· ${S.fyi} for information</span>` : ''}</span></div>
    ${st.running ? `<div style="margin-top:10px"><b>Running: ${esc(st.job)}…</b></div>` : buttons}
    ${st.log ? `<pre style="max-height:220px;overflow:auto;background:#fff;border:1px solid var(--line);padding:8px;font-size:12px;margin:10px 0 0">${esc(st.log)}</pre>` : ''}</div>`;
  if (st.running) { setTimeout(poll, 2000); window._wasRunning = true; }
  else if (window._wasRunning) location.reload();
}
if (APP) { poll(); document.querySelector('header b').insertAdjacentHTML('beforebegin', '<a href="/">All papers</a>'); }
count();
</script></body></html>"""


def main(paper_dir: str):
    path = f'{paper_dir}/review.html'
    open(path, 'w').write(build(paper_dir))
    print(f'-> {path}  (open it in a browser; Export corrections -> save corrections.json in {paper_dir})')


if __name__ == '__main__':
    main(sys.argv[1])
