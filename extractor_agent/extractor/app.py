"""app.py -- the extractor as a local web app: add papers, run each step, review, build the shared KG.

    export LLM_API_KEY=...             (default model gpt-6-luna on OpenAI; see extractor/llm.py)
    python -m extractor.app            # then open http://127.0.0.1:8765

Start page
    Papers       every paper folder, where it stands (OCR, boundary, symbols, laws, Sup description, graph),
                 how many items a person must check, and the next step. Tick papers to act on several.
    Add a paper  1. name it (author_year) and upload the PDF; 2. open the OCR notebook in Colab (link set
                 once on this page) and run it; 3. upload paddle.json and paddle_blocks.json. Equation
                 equation numbering then runs by itself (code only, no LLM).
    Selected     run extraction (symbols, laws, Sup description, graph) for the ticked papers whose boundary
                 is approved; rebuild their graphs (no LLM); build the shared KG from them.
    Shared KG    kg/shared_graph.ttl (the papers) and kg/shared_graph_with_schema.ttl (plus the ontology,
                 ready to import into GraphDB), with download links.
Paper page: review.html with a step panel on top. Every step is its own button, and approving the
boundary does not start the extraction. Items a person must check carry a 'resolved' box; inferred
items (FYI) do not.

Runs on this computer only (127.0.0.1); one job at a time. Each step is the same command as on the
command line, so its output is shown as the log.
"""
from __future__ import annotations
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from extractor import llm, review, triage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PAPERS = os.environ.get('APP_PAPERS') or os.path.join(ROOT, 'papers')
KG_DIR = os.path.join(ROOT, 'kg')
CONFIG = os.path.join(ROOT, 'app_config.json')
PORT = int(os.environ.get('APP_PORT', 8765))
JOB = {'running': False, 'paper': None, 'job': None, 'log': ''}
LOCK = threading.Lock()
ID = re.compile(r'^[a-z][a-z0-9_]{2,60}$')
UPLOADS = {'paper.pdf', 'paddle.json', 'paddle_blocks.json'}


def config() -> dict:
    return json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}


# ----------------------------------------------------------------------------- where a paper stands
def papers() -> list[str]:
    out = []
    for pid in sorted(os.listdir(PAPERS)):
        d = os.path.join(PAPERS, pid)
        if os.path.isdir(d) and ID.match(pid) and any(os.path.exists(os.path.join(d, f)) for f in UPLOADS):
            out.append(pid)
    return out


def equation_crop(paper_dir: str, eq: str) -> bytes | None:
    """The equation as printed in the PDF (PNG), from the position the OCR recorded: for the reviewer to tell an
    OCR misread from an error in the paper. Parts of a block (6.1) show the whole block. None if there is no position
    (an equation read from a table) or no PDF."""
    import fitz
    pdf = os.path.join(paper_dir, 'paper.pdf')
    if not os.path.exists(pdf):
        return None
    formulas = json.load(open(os.path.join(paper_dir, 'paddle.json')))['formulas']
    from extractor.paper import load_equations
    eqs, _ = load_equations(paper_dir)
    f = next((x for x in eqs if str(x.get('eq')) == eq), None)
    if f is None or not f.get('bbox') or f.get('source', 'main') != 'main' and not str(eq).startswith('C-'):
        return None
    dpi = json.load(open(os.path.join(paper_dir, 'paddle.json'))).get('dpi', 150)
    x0, y0, x1, y1 = [v * 72 / dpi for v in f['bbox']]
    doc = fitz.open(pdf)
    if f.get('page', 0) >= len(doc):
        return None
    clip = fitz.Rect(x0 - 6, y0 - 4, x1 + 30, y1 + 4)
    return doc[f['page']].get_pixmap(dpi=200, clip=clip).tobytes('png')


def stages(pid: str) -> dict:
    d = os.path.join(PAPERS, pid)
    has = lambda f: os.path.exists(os.path.join(d, f))
    mtime = lambda f: os.path.getmtime(os.path.join(d, f)) if has(f) else 0
    cfg = json.load(open(os.path.join(d, 'paper.json'))) if has('paper.json') else None
    boundary = 'none' if cfg is None else 'waiting' if cfg.get('status') == 'proposed' else 'done'
    newest_input = max(mtime('symbols.json'), mtime('laws.json'), mtime('sup_meta.json'), mtime('corrections.json'),
                       mtime('paper.json'))
    t = triage.summary(d) if has('symbols.json') or has('laws.json') else {'check': 0, 'resolved': 0, 'fyi': 0}
    ocr = 'done' if has('paddle.json') and has('paddle_blocks.json') else 'waiting' if has('paper.pdf') else 'none'
    return {'pdf': 'done' if has('paper.pdf') else 'none', 'ocr': ocr, 'boundary': boundary,
            'symbols': 'done' if has('symbols.json') else 'none', 'laws': 'done' if has('laws.json') else 'none',
            'supmeta': 'done' if has('sup_meta.json') else 'none',
            'graph': 'done' if has('sup_graph_pipeline.ttl') and mtime('sup_graph_pipeline.ttl') >= newest_input
            else 'stale' if has('sup_graph_pipeline.ttl') else 'none',
            'pending': t['check'], 'resolved': t['resolved'], 'fyi': t['fyi'],
            'title': re.sub(r'\s*\$[^$]*\$', '', (cfg or {}).get('title') or _title(d)).strip()}   # no '$^{\dagger}$'


def _title(d: str) -> str:
    p = os.path.join(d, 'paddle_blocks.json')
    if not os.path.exists(p):
        return ''
    return next((b['content'] for b in json.load(open(p))['blocks'] if b['label'] == 'doc_title'), '')


def next_step(s: dict) -> str:
    if s['ocr'] != 'done':
        return 'upload the PDF' if s['pdf'] == 'none' else 'run OCR in Colab, upload the two files'
    if s['boundary'] == 'none':
        return 'propose boundary'
    if s['boundary'] == 'waiting':
        return 'check and approve the boundary'
    if 'none' in (s['symbols'], s['laws'], s['supmeta']):
        return 'run extraction'
    if s['pending']:
        return f"review {s['pending']} item{'s' * (s['pending'] > 1)}"
    if s['graph'] != 'done':
        return 'rebuild graph'
    return 'done'


# ----------------------------------------------------------------------------- jobs
def _cmd(pid: str, step: str, *flags) -> list[str]:
    return [sys.executable, '-m', f'extractor.{step}', os.path.join('papers', pid), *flags]


def _run(cmds: list[list[str]]) -> bool:
    for c in cmds:
        JOB['log'] += f"\n$ {' '.join(c[2:])}\n"
        p = subprocess.Popen(c, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        for line in p.stdout:
            JOB['log'] += line
        if p.wait() != 0:
            JOB['log'] += f'\nstopped: {c[2]} failed (exit {p.returncode})\n'
            return False
    return True


EXTRACT = ['symbols', 'laws', 'fixes', 'supmeta', 'graph']
SINGLE = {'sup': [('sup',)], 'approve_only': [('sup', '--approve')],
          'approve': [('sup', '--approve')] + [(s,) for s in EXTRACT],
          'symbols': [('symbols',), ('graph',)], 'laws': [('laws',), ('fixes',), ('graph',)], 'supmeta': [('supmeta',), ('graph',)],
          'extract': [(s,) for s in EXTRACT], 'graph': [('graph',)],
          'prepare': [('numbering',)],
          'estimate': [('symbols', '--dry-run'), ('laws', '--dry-run'), ('supmeta', '--dry-run')]}


def job(pid: str, name: str, selected: list[str]):
    try:
        if pid != '*':
            _run([_cmd(pid, *step) for step in SINGLE[name]])
            return
        if name in ('batch_extract', 'batch_graph'):           # several papers, one after the other
            for p in selected:
                if stages(p)['boundary'] != 'done':
                    JOB['log'] += f'\n{p}: skipped, boundary not approved\n'
                    continue
                JOB['log'] += f'\n=== {p}\n'
                _run([_cmd(p, *step) for step in (SINGLE['extract'] if name == 'batch_extract' else SINGLE['graph'])])
        elif name == 'kg':
            build_kg(selected)
    finally:
        JOB['running'] = False


def build_kg(selected: list[str]):
    """The shared KG from the ticked papers: their Sup graphs side by side (assemble.py), and the same
    with the ontology added, to import into GraphDB in one file."""
    from rdflib import Graph
    from extractor.assemble import assemble
    ready = [p for p in selected if os.path.exists(os.path.join(PAPERS, p, 'sup_graph_pipeline.ttl'))]
    for p in sorted(set(selected) - set(ready)):
        JOB['log'] += f'{p}: no graph yet, left out\n'
    if not ready:
        JOB['log'] += 'nothing to build\n'
        return
    shared, report = assemble([os.path.join(PAPERS, p) for p in ready])
    os.makedirs(KG_DIR, exist_ok=True)
    try:                                                   # the Sup map: embeddings are cached, so only new Sups cost
        from extractor import supmap
        smap, mdata = supmap.build(shared)
        mdata['clusters'] = {str(k): v for k, v in mdata.get('clusters', {}).items()}
        json.dump(mdata, open(os.path.join(KG_DIR, 'sup_map.json'), 'w'), indent=1, ensure_ascii=False)
        smap.serialize(os.path.join(KG_DIR, 'sup_map.ttl'), format='turtle')
        supmap.picture(mdata, os.path.join(KG_DIR, 'sup_map.png'))
        shared += smap
        JOB['log'] += f"Sup map: {len(mdata['sups'])} Sups in {len(mdata['clusters'])} groups -> kg/sup_map.png\n"
        from extractor import links                        # described edges between related Sups (LLM, cached)
        lgraph, lrep = links.build(shared, mdata)
        lgraph.serialize(os.path.join(KG_DIR, 'sup_links.ttl'), format='turtle')
        json.dump(lrep, open(os.path.join(KG_DIR, 'sup_links.json'), 'w'), indent=1)
        shared += lgraph
        JOB['log'] += (f"Links: {lrep['pairs']} pairs, {lrep['linked']} linked, {lrep['not_related']} not related "
                       f"({lrep['kept']} kept, {lrep['asked']} asked, {lrep['by_hand']} by hand); {len(lrep['cites'])} cites "
                       f"({llm.usage_line()})\n")
    except (SystemExit, Exception) as e:
        JOB['log'] += f'Sup map or links not built ({e}); the shared KG is built without them\n'
    shared.serialize(os.path.join(KG_DIR, 'shared_graph.ttl'), format='turtle')
    with_schema = Graph().parse(os.path.join(ROOT, 'schema', 'kg_schema.ttl')) + shared
    for prefix, ns in shared.namespaces():
        with_schema.bind(prefix, ns)
    with_schema.serialize(os.path.join(KG_DIR, 'shared_graph_with_schema.ttl'), format='turtle')
    report['papers'] = ready
    report['built'] = time.strftime('%Y-%m-%d %H:%M')
    json.dump(report, open(os.path.join(KG_DIR, 'shared_graph_report.json'), 'w'), indent=1, ensure_ascii=False)
    JOB['log'] += (f"shared KG: {len(ready)} papers, {len(report['sups'])} Sups, {report['triples']} triples, "
                   f"{len(report['schema_problems'])} schema problems\n-> kg/shared_graph.ttl, kg/shared_graph_with_schema.ttl\n")


def start(pid: str, name: str, selected=None) -> bool:
    if pid != '*' and name not in SINGLE:
        return False
    with LOCK:
        if JOB['running']:
            return False
        JOB.update(running=True, paper=pid, job=name, log='')
    threading.Thread(target=job, args=(pid, name, selected or []), daemon=True).start()
    return True


# ----------------------------------------------------------------------------- pages
STYLE = """body{font:15px system-ui,Arial;margin:0;color:#1d2a30;background:#f6f8f9} header{background:#0e3440;color:#fff;padding:12px 20px;display:flex;gap:16px;align-items:center}
main{max-width:1180px;margin:20px auto;padding:0 20px} table{border-collapse:collapse;width:100%;background:#fff}
td,th{border-bottom:1px solid #d9e0e3;padding:7px 8px;text-align:left;vertical-align:top} th{font-size:13px;color:#4b5b62}
a{color:#16706d} .card{background:#fff;border:1px solid #d9e0e3;border-radius:6px;padding:14px 16px;margin:14px 0}
.ok{color:#1b7f3b} .wait{color:#b26b00} .none{color:#9aa7ad} .mut{color:#5f6f76;font-size:13px} .title{color:#5f6f76;font-size:12px;max-width:300px}
button{font:inherit;padding:5px 12px;border:1px solid #9fb3ba;border-radius:5px;background:#fff;cursor:pointer} button.main{background:#0e3440;color:#fff;border-color:#0e3440}
button:disabled{opacity:.5;cursor:default} pre{max-height:260px;overflow:auto;background:#fff;border:1px solid #d9e0e3;padding:8px;font-size:12px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px} .steps .card{margin:0} h2{font-size:18px;margin:26px 0 6px}
input[type=text]{font:inherit;padding:4px 8px;border:1px solid #9fb3ba;border-radius:5px} .next{font-weight:600}"""

INDEX_JS = """
const sel = () => [...document.querySelectorAll('.sel:checked')].map(x => x.value);
const all_ = v => { document.querySelectorAll('.sel').forEach(x => x.checked = v); upd(); };
const upd = () => document.getElementById('nsel').textContent = sel().length + ' selected';
document.addEventListener('change', e => e.target.classList.contains('sel') && upd()); upd();
const esc = s => (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function batch(job) { if (!sel().length) return alert('tick at least one paper');
  const r = await fetch('/api/*/run', {method: 'POST', body: JSON.stringify({job, selected: sel()})});
  if (!r.ok) alert('another job is running'); poll(); }
async function poll() { const st = await (await fetch('/api/*/status')).json(); const el = document.getElementById('job');
  el.innerHTML = st.log ? `<div class="card"><b>${st.running ? 'Running: ' + st.job + '…' : 'Last job: ' + st.job}</b><pre>${esc(st.log)}</pre></div>` : '';
  if (st.running) { window._r = 1; setTimeout(poll, 2000); } else if (window._r) location.reload(); }
async function addPaper() { const id = document.getElementById('newid').value.trim(), f = document.getElementById('pdf').files[0], m = document.getElementById('addmsg');
  if (!/^[a-z][a-z0-9_]{2,60}$/.test(id)) return m.textContent = 'name: lower case letters, digits, _';
  if (!f) return m.textContent = 'choose the PDF';
  let r = await fetch('/api/new', {method: 'POST', body: JSON.stringify({id})}); if (!r.ok) return m.textContent = await r.text();
  r = await fetch(`/api/${id}/upload?file=paper.pdf`, {method: 'POST', body: f});
  m.textContent = r.ok ? 'added' : 'upload failed'; if (r.ok) location.reload(); }
async function uploadOcr() { const id = document.getElementById('upid').value, fs = [...document.getElementById('ocrfiles').files], m = document.getElementById('upmsg');
  if (!id || !fs.length) return m.textContent = 'choose the paper and the files';
  for (const f of fs) { const r = await fetch(`/api/${id}/upload?file=${encodeURIComponent(f.name)}`, {method: 'POST', body: f});
    if (!r.ok) return m.textContent = f.name + ': ' + await r.text(); }
  m.textContent = 'uploaded: equation numbering is running'; window._r = 1; poll(); }
async function saveColab() { await fetch('/api/config', {method: 'POST', body: JSON.stringify({colab_url: document.getElementById('colab').value.trim()})}); location.reload(); }
poll();
"""


def index() -> str:
    mark = lambda v: {'done': "<span class='ok'>✓</span>", 'waiting': "<span class='wait'>●</span>",
                      'stale': "<span class='wait' title='inputs changed since the graph was built'>↻</span>",
                      'none': "<span class='none'>○</span>"}[v]
    rows, waiting_ocr = [], []
    for pid in papers():
        s = stages(pid)
        if s['ocr'] != 'done':
            waiting_ocr.append(pid)
        n = next_step(s)
        rows.append(f"<tr><td><input type='checkbox' class='sel' value='{pid}' {'checked' if s['graph'] != 'none' else ''}></td>"
                    f"<td><a href='/paper/{pid}'>{pid}</a><div class='title'>{html.escape(s['title'][:90])}</div></td>"
                    + ''.join(f"<td>{mark(s[k])}</td>" for k in ('ocr', 'boundary', 'symbols', 'laws', 'supmeta', 'graph'))
                    + f"<td>{s['pending'] or ''}</td><td class='{'next' if n != 'done' else 'ok'}'>{n}</td></tr>")
    key = "<span class='ok'>set</span>" if llm.api_key() else "<b style='color:#b42318'>not set: export LLM_API_KEY before starting the app</b>"
    colab = config().get('colab_url', '')
    kg = os.path.join(KG_DIR, 'shared_graph_report.json')
    kgr = json.load(open(kg)) if os.path.exists(kg) else None
    kg_line = (f"Last built {kgr.get('built', '?')}: {len(kgr.get('papers', []))} papers, {len(kgr['sups'])} Sups, "
               f"{kgr['triples']} triples. <a href='/download/shared_graph_with_schema.ttl'>Download for GraphDB</a> "
               f"(with the ontology) · <a href='/download/shared_graph.ttl'>papers only</a>"
               + (" · <a href='/download/sup_map.png' target='_blank'>map of the Sups</a>" if os.path.exists(os.path.join(KG_DIR, 'sup_map.png')) else '')) if kgr else 'Not built yet.'
    colab_btn = (f"<a href='{html.escape(colab)}' target='_blank'><button class='main'>Open the OCR notebook</button></a>"
                 if colab else "<span class='mut'>No notebook link yet: paste it below.</span>")
    return f"""<!doctype html><meta charset="utf-8"><title>Extractor</title><style>{STYLE}</style>
<header><b>Extractor</b><span style="color:#cfe3ea;font-size:13px">papers in {html.escape(PAPERS)}</span></header><main>
<p class="mut">LLM key: {key} · model: {llm.MODEL} · effort: {llm.EFFORT}</p>
<h2>Papers</h2>
<div style="display:flex;gap:8px;align-items:center;margin:6px 0 8px;flex-wrap:wrap">
 <button onclick="all_(true)">Select all</button><button onclick="all_(false)">None</button>
 <span class="mut" id="nsel"></span><span style="margin-left:auto"></span>
 <button onclick="batch('batch_extract')" title="symbols, laws, Sup description and graph; papers without an approved boundary are skipped">Run extraction for selected</button>
 <button onclick="batch('batch_graph')">Rebuild graphs (no LLM)</button>
 <button class="main" onclick="batch('kg')">Build shared KG from selected</button></div>
<table><tr><th></th><th>Paper</th><th>OCR</th><th>Boundary</th><th>Symbols</th><th>Laws</th><th>Sup description</th><th>Graph</th><th>To check</th><th>Next step</th></tr>
{''.join(rows)}</table>
<p class="mut">✓ done · ● waiting for you · ↻ graph older than its inputs · ○ not yet</p>
<div class="card"><b>Shared knowledge graph</b> <span class="mut">{kg_line}</span></div>
<div id="job"></div>
<h2>Add a paper</h2>
<div class="steps">
 <div class="card"><b>1. Name it, upload the PDF</b><p class="mut">Name as author_year, lower case (nausch_2023).</p>
  <input type="text" id="newid" placeholder="author_year"><p><input type="file" id="pdf" accept=".pdf"></p>
  <button class="main" onclick="addPaper()">Add</button> <span id="addmsg" class="mut"></span></div>
 <div class="card"><b>2. Read it in Colab (GPU)</b><p class="mut">Put the PDF on Drive as thesis/papers/&lt;name&gt;.pdf (a supplementary
  file as &lt;name&gt;_supp.pdf), open the notebook, set BATCH = 'new', Run all.</p>{colab_btn}
  <p class="mut">Notebook link:<br><input type="text" id="colab" value="{html.escape(colab)}" placeholder="https://colab.research.google.com/..." style="width:95%"></p>
  <button onclick="saveColab()">Save link</button></div>
 <div class="card"><b>3. Upload what Colab wrote</b><p class="mut">From Drive thesis/papers/&lt;name&gt;/: paddle.json and paddle_blocks.json
  (both at once). Equation numbering then runs by itself.</p>
  {f"<select id='upid'>{''.join(f'<option>{p}</option>' for p in waiting_ocr)}</select>" if waiting_ocr else "<select id='upid' disabled><option value=''>no paper waiting for OCR</option></select>"}
  <p><input type="file" id="ocrfiles" accept=".json" multiple></p><button class="main" onclick="uploadOcr()">Upload</button> <span id="upmsg" class="mut"></span></div>
</div></main><script>{INDEX_JS}</script>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype='text/html; charset=utf-8', code=200, name=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        if name:
            self.send_header('Content-Disposition', f'attachment; filename="{name}"')
        self.end_headers()
        self.wfile.write(data)

    def _pid(self, part: str) -> str | None:
        return part if part and ID.match(part) and os.path.isdir(os.path.join(PAPERS, part)) else None

    def do_GET(self):
        url = urlparse(self.path)
        parts = url.path.strip('/').split('/')
        if url.path == '/':
            return self._send(index())
        if parts[0] == 'paper' and len(parts) == 2 and (pid := self._pid(parts[1])):
            if not os.path.exists(os.path.join(PAPERS, pid, 'paddle.json')):
                return self._send(f'<p>{pid}: no OCR output yet. Run the notebook and upload paddle.json and '
                                  f'paddle_blocks.json on the start page.</p><p><a href="/">back</a></p>')
            try:
                return self._send(review.build(os.path.join(PAPERS, pid), app=True))
            except (SystemExit, Exception) as e:                # show the problem instead of a blank page
                return self._send(f'<p>{type(e).__name__}: {html.escape(str(e))}</p><p><a href="/">back</a></p>', code=500)
        if parts[0] == 'crop' and len(parts) == 3 and (pid := self._pid(parts[1])):   # the PDF region of one equation
            png = equation_crop(os.path.join(PAPERS, pid), parts[2].removesuffix('.png'))
            return self._send(png, 'image/png') if png else self._send('no crop', 'text/plain', 404)
        if parts[0] == 'download' and len(parts) == 2 and parts[1] in ('shared_graph.ttl', 'shared_graph_with_schema.ttl', 'sup_map.png'):
            p = os.path.join(KG_DIR, parts[1])
            if os.path.exists(p):
                if parts[1].endswith('.png'):
                    return self._send(open(p, 'rb').read(), 'image/png')
                return self._send(open(p, 'rb').read(), 'text/turtle', name=parts[1])
        if parts[0] == 'api' and len(parts) == 3 and parts[2] == 'status' and (parts[1] == '*' or self._pid(parts[1])):
            pid = parts[1]
            mine = JOB['paper'] == pid
            return self._send(json.dumps({'running': JOB['running'] and mine, 'busy': JOB['running'], 'job': JOB['job'],
                                          'log': JOB['log'][-8000:] if mine else '',
                                          'stages': stages(pid) if pid != '*' else None}), 'application/json')
        self._send('not found', 'text/plain', 404)

    def do_POST(self):
        url = urlparse(self.path)
        parts = url.path.strip('/').split('/')
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)) or 0)
        if parts == ['api', 'config']:
            json.dump({**config(), **json.loads(body)}, open(CONFIG, 'w'), indent=1)
            return self._send('{"ok": true}', 'application/json')
        if parts == ['api', 'new']:
            pid = json.loads(body).get('id', '')
            if not ID.match(pid):
                return self._send('name: lower case letters, digits and _', 'text/plain', 400)
            if os.path.exists(os.path.join(PAPERS, pid, 'paper.pdf')):
                return self._send(f'{pid} already exists', 'text/plain', 409)
            os.makedirs(os.path.join(PAPERS, pid), exist_ok=True)
            return self._send('{"ok": true}', 'application/json')
        if parts == ['api', '*', 'run']:
            req = json.loads(body)
            sel = [p for p in req.get('selected', []) if self._pid(p)]
            ok = req.get('job') in ('batch_extract', 'batch_graph', 'kg') and start('*', req['job'], sel)
            return self._send(json.dumps({'started': ok}), 'application/json', 200 if ok else 409)
        pid = self._pid(parts[1]) if len(parts) == 3 and parts[0] == 'api' else None
        if not pid:
            return self._send('not found', 'text/plain', 404)
        d = os.path.join(PAPERS, pid)
        if parts[2] == 'upload':
            name = os.path.basename(parse_qs(url.query).get('file', [''])[0])
            name = 'paper.pdf' if name.lower().endswith('.pdf') else name
            if name not in UPLOADS:
                return self._send('expected a PDF, paddle.json or paddle_blocks.json', 'text/plain', 400)
            if name.endswith('.json'):
                try:
                    key = 'formulas' if name == 'paddle.json' else 'blocks'
                    if key not in json.loads(body):
                        return self._send(f'has no "{key}": not a file from the OCR notebook?', 'text/plain', 400)
                except ValueError:
                    return self._send('not valid JSON', 'text/plain', 400)
            open(os.path.join(d, name), 'wb').write(body)
            if name.endswith('.json') and all(os.path.exists(os.path.join(d, f)) for f in ('paddle.json', 'paddle_blocks.json')):
                start(pid, 'prepare')                         # equation numbering: code only
            return self._send('{"ok": true}', 'application/json')
        if parts[2] == 'corrections':
            corr = json.loads(body)
            if any(corr.get(k) for k in ('sup', 'symbols', 'laws', 'new_models')) or os.path.exists(os.path.join(d, 'corrections.json')):
                json.dump(corr, open(os.path.join(d, 'corrections.json'), 'w'), indent=1, ensure_ascii=False)
            return self._send('{"ok": true}', 'application/json')
        if parts[2] == 'run':
            ok = start(pid, json.loads(body).get('job'))
            return self._send(json.dumps({'started': ok}), 'application/json', 200 if ok else 409)
        self._send('not found', 'text/plain', 404)

    def log_message(self, *a):
        pass


def main():
    if not llm.api_key():
        print('note: no LLM_API_KEY set; review works, the LLM steps will fail')
    print(f'extractor app: http://127.0.0.1:{PORT}   (Ctrl+C to stop)')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()


if __name__ == '__main__':
    main()
