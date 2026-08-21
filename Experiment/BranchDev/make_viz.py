"""Build a self-contained interactive page for reading the embedding space.

The page exists to answer one question by eye: when two continuations land close
together, are they close because they take the same mathematical approach, or
because the same model wrote them? A static scatter cannot answer that, so every
point carries its full text and the page lets you pull points up and read them.

No clustering is applied. Projection is PCA, which has no tunable parameters and
therefore cannot manufacture structure that is not in the data.
"""
import argparse, html, json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def pca2(X):
    """Two leading principal components, computed by SVD. No knobs."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    var = (S ** 2) / (S ** 2).sum()
    return Y, float(var[0]), float(var[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(OUT, "emb_qwen3emb8b_raw.npy"))
    ap.add_argument("--index", default=os.path.join(OUT, "emb_qwen3emb8b_raw_index.jsonl"))
    ap.add_argument("--source", default=os.path.join(OUT, "discovery_pilot_v2.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "embedding_explorer.html"))
    a = ap.parse_args()

    X = np.load(a.emb)
    idx = [json.loads(l) for l in open(a.index)]
    src = [json.loads(l) for l in open(a.source)]
    src = [r for r in src if "error" not in r and (r.get("reasoning") or "").strip()]
    assert len(idx) == len(src) == len(X), (len(idx), len(src), len(X))

    problems = {}
    for r in src:
        problems.setdefault(r["problem_id"], {
            "level": r.get("level"), "type": r.get("type"), "answer": r.get("answer")})
    # problem statements
    import sys
    sys.path.insert(0, ROOT)
    import or_common
    stmt = {p["id"]: p["problem"] for p in or_common.load_problems()}

    per_problem = {}
    for pid in sorted(problems):
        sel = [i for i, m in enumerate(idx) if m["problem_id"] == pid]
        Y, v1, v2 = pca2(X[sel])
        Xn = X[sel]
        sim = Xn @ Xn.T                      # cosine, vectors are normalised
        np.fill_diagonal(sim, -np.inf)
        pts = []
        for j, i in enumerate(sel):
            nb = int(np.argmax(sim[j]))
            pts.append({
                "x": round(float(Y[j, 0]), 4), "y": round(float(Y[j, 1]), 4),
                "m": src[i]["model"].split("/")[-1],
                "k": src[i]["sample_k"],
                "t": src[i]["reasoning"],
                "c": src[i].get("content") or "",
                "nn": j if nb < 0 else int(nb),
                "ns": round(float(sim[j, nb]), 4) if np.isfinite(sim[j, nb]) else 0.0,
            })
        # summary stats that need no thresholds
        off = sim.copy()
        finite = off[np.isfinite(off)]
        same, diff = [], []
        for p in range(len(sel)):
            for q in range(p + 1, len(sel)):
                (same if src[sel[p]]["model"] == src[sel[q]]["model"] else diff).append(sim[p, q])
        per_problem[pid] = {
            "pts": pts, "v1": round(v1, 3), "v2": round(v2, 3),
            "level": problems[pid]["level"], "type": problems[pid]["type"],
            "answer": problems[pid]["answer"], "stmt": stmt.get(pid, ""),
            "sim_all": round(float(np.mean(finite)), 4),
            "sim_same": round(float(np.mean(same)), 4),
            "sim_diff": round(float(np.mean(diff)), 4),
            "nn_same_pct": round(100.0 * sum(
                1 for j, p in enumerate(pts) if p["m"] == pts[p["nn"]]["m"]) / len(pts), 1),
        }

    models = sorted({p["m"] for d in per_problem.values() for p in d["pts"]})
    payload = {"problems": {str(k): v for k, v in per_problem.items()}, "models": models}

    tpl = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    with open(a.out, "w") as f:
        f.write(tpl)
    size = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}  ({size:.1f} MB, {len(idx)} points, {len(per_problem)} problems)")
    print("\nper-problem summary (no thresholds applied):")
    print(f"{'pid':>4}{'lvl':>9}{'PC1+PC2':>10}{'mean cos':>10}{'same-src':>10}"
          f"{'diff-src':>10}{'gap':>8}{'NN same src %':>15}")
    for pid, d in sorted(per_problem.items()):
        gap = d["sim_same"] - d["sim_diff"]
        print(f"{pid:>4}{str(d['level'])[-1:]:>9}{d['v1']+d['v2']:>10.2f}{d['sim_all']:>10.3f}"
              f"{d['sim_same']:>10.3f}{d['sim_diff']:>10.3f}{gap:>8.3f}{d['nn_same_pct']:>15.1f}")
    gaps = [d["sim_same"] - d["sim_diff"] for d in per_problem.values()]
    nns = [d["nn_same_pct"] for d in per_problem.values()]
    print(f"\nmean same-source minus diff-source cosine: {np.mean(gaps):+.3f}")
    print(f"mean nearest-neighbour-is-same-source: {np.mean(nns):.1f}%  "
          f"(chance = {100.0/len(models):.1f}% if style were irrelevant)")


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Branch discovery embedding explorer</title>
<style>
:root{--ink:#1F2937;--muted:#6B7280;--grid:#E5E7EB;--bg:#fff;--panel:#F9FAFB}
*{box-sizing:border-box}
body{margin:0;font:13.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     color:var(--ink);background:var(--bg)}
header{padding:14px 18px;border-bottom:1px solid var(--grid);display:flex;gap:18px;
       align-items:center;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:15px;margin:0;font-weight:600}
select,button{font:inherit;padding:5px 9px;border:1px solid var(--grid);border-radius:6px;
              background:#fff;color:var(--ink)}
button{cursor:pointer}
button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.wrap{display:grid;grid-template-columns:minmax(420px,1fr) minmax(360px,520px);gap:0;
      height:calc(100vh - 58px)}
#plot{position:relative;border-right:1px solid var(--grid)}
canvas{display:block;width:100%;height:100%}
#side{overflow-y:auto;padding:14px 16px;background:var(--panel)}
.stat{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);
      padding:8px 18px;border-bottom:1px solid var(--grid);background:var(--panel)}
.stat b{color:var(--ink);font-weight:600}
.legend{display:flex;gap:10px;flex-wrap:wrap;align-items:center;font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
.legend i{width:10px;height:10px;border-radius:50%;display:inline-block}
.legend .off{opacity:.28}
.card{background:#fff;border:1px solid var(--grid);border-radius:8px;padding:11px 13px;margin-bottom:11px}
.card h3{margin:0 0 6px;font-size:12.5px;display:flex;justify-content:space-between;align-items:center}
.tag{font-size:11px;color:#fff;padding:1px 7px;border-radius:99px}
.txt{white-space:pre-wrap;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
     max-height:290px;overflow:auto;background:#FCFCFD;border:1px solid var(--grid);
     border-radius:6px;padding:8px}
.q{background:#fff;border:1px solid var(--grid);border-left:3px solid var(--ink);
   border-radius:6px;padding:10px 12px;margin-bottom:12px;font-size:12.5px}
.hint{color:var(--muted);font-size:12px;margin:6px 0 12px}
.pair{font-size:11.5px;color:var(--muted);margin-top:6px}
</style>
<header>
  <h1>Branch discovery &mdash; embedding space</h1>
  <label>problem
    <select id="pid"></select>
  </label>
  <button id="pairbtn">show nearest-neighbour links</button>
  <span class="legend" id="legend"></span>
</header>
<div class="stat" id="stat"></div>
<div class="wrap">
  <div id="plot"><canvas id="cv"></canvas></div>
  <div id="side"></div>
</div>
<script>
const DATA = __DATA__;
const PALETTE = ["#2563EB","#EA580C","#059669","#7C3AED","#DB2777","#0891B2","#CA8A04","#DC2626"];
const colorOf = {};
DATA.models.forEach((m,i)=>colorOf[m]=PALETTE[i%PALETTE.length]);
const hidden = new Set();
let pid = Object.keys(DATA.problems)[0];
let showPairs = false, sel = null, hover = null;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let tf = {sx:1,sy:1,ox:0,oy:0};

const sels = document.getElementById('pid');
Object.keys(DATA.problems).sort((a,b)=>a-b).forEach(k=>{
  const d = DATA.problems[k];
  const o = document.createElement('option');
  o.value = k; o.textContent = `${k}  ${d.level||''}  ${d.type||''}`;
  sels.appendChild(o);
});
sels.value = pid;
sels.onchange = e => { pid = e.target.value; sel = null; draw(); side(); stat(); };
document.getElementById('pairbtn').onclick = e => {
  showPairs = !showPairs; e.target.classList.toggle('on', showPairs); draw();
};

function legend(){
  const el = document.getElementById('legend'); el.innerHTML='';
  DATA.models.forEach(m=>{
    const s=document.createElement('span');
    s.innerHTML=`<i style="background:${colorOf[m]}"></i>${m}`;
    s.className = hidden.has(m)?'off':'';
    s.onclick=()=>{ hidden.has(m)?hidden.delete(m):hidden.add(m); legend(); draw(); };
    el.appendChild(s);
  });
}
function stat(){
  const d = DATA.problems[pid];
  document.getElementById('stat').innerHTML =
    `<span>points <b>${d.pts.length}</b></span>`+
    `<span>PC1+PC2 variance <b>${(100*(d.v1+d.v2)).toFixed(0)}%</b></span>`+
    `<span>mean cosine <b>${d.sim_all}</b></span>`+
    `<span>same-source <b>${d.sim_same}</b></span>`+
    `<span>different-source <b>${d.sim_diff}</b></span>`+
    `<span>gap <b>${(d.sim_same-d.sim_diff>=0?'+':'')}${(d.sim_same-d.sim_diff).toFixed(3)}</b></span>`+
    `<span>nearest neighbour is same source <b>${d.nn_same_pct}%</b> (chance ${(100/DATA.models.length).toFixed(0)}%)</span>`;
}
function resize(){
  const r = cv.parentElement.getBoundingClientRect();
  cv.width = r.width*devicePixelRatio; cv.height = r.height*devicePixelRatio;
  cv.style.width=r.width+'px'; cv.style.height=r.height+'px';
  draw();
}
function draw(){
  const d = DATA.problems[pid], P = d.pts;
  const W = cv.width, H = cv.height, pad = 46*devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const xs=P.map(p=>p.x), ys=P.map(p=>p.y);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  tf.sx=(W-2*pad)/((x1-x0)||1); tf.sy=(H-2*pad)/((y1-y0)||1);
  tf.ox=pad-x0*tf.sx; tf.oy=pad-y0*tf.sy;
  const X=p=>p.x*tf.sx+tf.ox, Y=p=>H-(p.y*tf.sy+tf.oy);
  ctx.strokeStyle='#E5E7EB'; ctx.lineWidth=devicePixelRatio;
  ctx.strokeRect(pad/2,pad/2,W-pad,H-pad);
  if(showPairs){
    ctx.strokeStyle='rgba(107,114,128,.32)'; ctx.lineWidth=devicePixelRatio;
    P.forEach((p,i)=>{ const q=P[p.nn]; if(!q) return;
      if(hidden.has(p.m)||hidden.has(q.m)) return;
      ctx.beginPath(); ctx.moveTo(X(p),Y(p)); ctx.lineTo(X(q),Y(q)); ctx.stroke(); });
  }
  P.forEach((p,i)=>{
    if(hidden.has(p.m)) return;
    const r=(i===sel?7:i===hover?6:4)*devicePixelRatio;
    ctx.beginPath(); ctx.arc(X(p),Y(p),r,0,7);
    ctx.fillStyle=colorOf[p.m]; ctx.globalAlpha=(i===sel||i===hover)?1:.72; ctx.fill();
    ctx.globalAlpha=1;
    if(i===sel){ ctx.strokeStyle='#111'; ctx.lineWidth=2*devicePixelRatio; ctx.stroke(); }
  });
  ctx.fillStyle='#6B7280'; ctx.font=`${11*devicePixelRatio}px sans-serif`;
  ctx.fillText(`PC1 ${(100*d.v1).toFixed(0)}%`, W-pad-70*devicePixelRatio, H-pad/2+12*devicePixelRatio);
  ctx.save(); ctx.translate(pad/2-14*devicePixelRatio, pad+40*devicePixelRatio);
  ctx.rotate(-Math.PI/2); ctx.fillText(`PC2 ${(100*d.v2).toFixed(0)}%`,0,0); ctx.restore();
}
function pick(ev){
  const d=DATA.problems[pid], P=d.pts, H=cv.height;
  const r=cv.getBoundingClientRect();
  const mx=(ev.clientX-r.left)*devicePixelRatio, my=(ev.clientY-r.top)*devicePixelRatio;
  let best=null,bd=1e9;
  P.forEach((p,i)=>{ if(hidden.has(p.m)) return;
    const dx=(p.x*tf.sx+tf.ox)-mx, dy=(H-(p.y*tf.sy+tf.oy))-my, dd=dx*dx+dy*dy;
    if(dd<bd){bd=dd;best=i;} });
  return bd < (14*devicePixelRatio)**2 ? best : null;
}
cv.onmousemove = e => { const h=pick(e); if(h!==hover){hover=h;draw();
  cv.style.cursor = h===null?'default':'pointer';} };
cv.onclick = e => { const h=pick(e); if(h!==null){ sel=h; draw(); side(); } };

function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function card(p,i,label){
  return `<div class="card"><h3><span>${label} &nbsp;<span class="tag" style="background:${colorOf[p.m]}">${p.m}</span> &nbsp;sample ${p.k}</span></h3>`+
         `<div class="txt">${esc(p.t)}</div>`+
         (p.c?`<div class="pair">final answer: ${esc(p.c).slice(0,200)}</div>`:'')+`</div>`;
}
function side(){
  const d=DATA.problems[pid], P=d.pts, el=document.getElementById('side');
  let h = `<div class="q"><b>problem ${pid}</b> &middot; ${d.level||''} &middot; ${d.type||''} &middot; answer <code>${esc(d.answer)}</code><br><br>${esc(d.stmt)}</div>`;
  if(sel===null){
    h += `<div class="hint">Click any point to read its reasoning text. Then click its nearest neighbour (shown below it) and judge for yourself whether the two use the same mathematical approach, or merely the same writing style.</div>`;
    // seed the panel with the closest and the farthest pair
    let bi=0,bs=-2,wi=0,ws=2;
    P.forEach((p,i)=>{ if(p.ns>bs){bs=p.ns;bi=i;} if(p.ns<ws){ws=p.ns;wi=i;} });
    h += `<div class="hint">Seeded with the tightest and the loosest pair in this problem.</div>`;
    h += card(P[bi],bi,`tightest pair &mdash; cosine ${bs}`);
    h += card(P[P[bi].nn],P[bi].nn,'its nearest neighbour');
    h += card(P[wi],wi,`most isolated point &mdash; best cosine only ${ws}`);
    h += card(P[P[wi].nn],P[wi].nn,'its nearest neighbour');
  } else {
    const p=P[sel], q=P[p.nn];
    h += card(p,sel,'selected');
    h += `<div class="hint">Its nearest neighbour in the full embedding space (cosine ${p.ns}) &mdash; same approach, or same style?</div>`;
    h += card(q,p.nn,'nearest neighbour');
  }
  el.innerHTML=h; el.scrollTop=0;
}
legend(); stat(); side(); resize();
addEventListener('resize', resize);
</script>
"""

if __name__ == "__main__":
    main()
