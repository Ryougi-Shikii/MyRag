"""Interactive pipeline graph, rendered as one self-contained HTML page.

Why vis-network (the engine behind pyvis) embedded through components.html,
and why not pyvis itself, streamlit-agraph or Plotly:
  * the detail panel next to the graph and the level-1 -> level-2 expansion are
    client-side: all turn traces are embedded as JSON, so clicking a node needs
    no Streamlit rerun (a rerun would reload the iframe and lose the view);
  * per-node styling (glow, dashed borders, rounded boxes, opacity fade),
    fixed layered layout and arrowed bezier edges are first-class in vis-network;
  * pyvis only wraps it and hides the click hooks; streamlit-agraph has the same
    limits; Plotly cannot draw arrowed, clickable node graphs well.
The only external resource is the vis-network script from unpkg.
"""
from __future__ import annotations

import json
from typing import Any

VIS_URL = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="__VIS_URL__"></script>
<style>
  :root { --bg:#0e1117; --card:#161b22; --line:#2d333b; --text:#e6edf3; --muted:#8b949e;
          --routing:#f59e0b; --planning:#8b5cf6; --retrieval:#3b82f6; --rerank:#22c55e; --answer:#ec4899; }
  * { box-sizing: border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
               font:13px/1.45 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
  #wrap { display:flex; height:100%; gap:10px; padding:8px; }
  #main { flex:1 1 66%; min-width:0; display:flex; flex-direction:column; gap:8px; }
  #legend { display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; color:var(--muted); font-size:12px; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:5px; vertical-align:-1px; }
  .live { color:#f87171; font-weight:600; animation:pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity:.35; } }
  #net { flex:1 1 auto; min-height:260px; border:1px solid var(--line); border-radius:12px;
         background:radial-gradient(circle at 50% 0%, #131a24 0%, var(--bg) 70%); }
  #timing { flex:0 0 auto; max-height:190px; overflow:auto; border:1px solid var(--line); border-radius:12px;
            background:var(--card); padding:8px 10px; }
  #timing h4, #panel h4 { margin:0 0 6px; font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }
  .trow { display:grid; grid-template-columns:118px 1fr 62px; gap:8px; align-items:center; height:17px; }
  .trow .bar { height:9px; border-radius:5px; min-width:2px; }
  .trow .nm { color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .trow .ms { color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }
  #panel { flex:0 0 33%; min-width:260px; overflow:auto; border:1px solid var(--line); border-radius:12px;
           background:var(--card); padding:12px 14px; }
  .pill { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11px; border:1px solid var(--line); color:var(--muted); margin-right:6px; }
  .pill.ok { color:#3fb950; border-color:#238636; } .pill.failed { color:#f85149; border-color:#da3633; }
  .pill.skipped, .pill.cancelled, .pill.pending { color:#8b949e; } .pill.running { color:#f59e0b; border-color:#9e6a03; }
  .badge { display:inline-block; padding:1px 8px; border-radius:6px; font-weight:600; font-size:12px; color:#0e1117; margin-right:6px; }
  .err { background:#3b1219; border:1px solid #da3633; color:#ffd7d5; border-radius:8px; padding:8px 10px; margin:8px 0; white-space:pre-wrap; }
  .sec { margin:12px 0 4px; font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  pre, .box { background:#0d1117; border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin:0;
              white-space:pre-wrap; word-break:break-word; font:12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  .chip { display:inline-block; padding:0 7px; margin:0 4px 3px 0; border-radius:999px; background:#1f2937; border:1px solid #374151; font-size:11px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  td, th { padding:4px 6px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
  th { color:var(--muted); font-weight:500; }
  .kv { display:grid; grid-template-columns:auto 1fr; gap:3px 12px; }
  .kv div:nth-child(odd) { color:var(--muted); }
  .hint { color:var(--muted); font-size:12px; margin-top:10px; }
  .miniBar { height:6px; border-radius:3px; background:#22c55e; margin-top:3px; }
  details > summary { cursor:pointer; color:var(--muted); }
  #nolib { padding:30px; color:#f87171; }
</style></head>
<body>
<div id="wrap">
  <div id="main">
    <div id="legend"></div>
    <div id="net"></div>
    <div id="timing"></div>
  </div>
  <div id="panel"></div>
</div>
<script>
const DATA = __DATA__;
const FAM = {routing:'#f59e0b', planning:'#8b5cf6', retrieval:'#3b82f6', rerank:'#22c55e', answer:'#ec4899'};
const FAM_NAMES = {routing:'Routing', planning:'Planning', retrieval:'Retrieval', rerank:'Rerank', answer:'Answer'};
const ACTIVE = new Set(['ok','running','failed']);

function esc(s){ return String(s==null?'':s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmtMs(ms){ if(ms==null) return ''; return ms<1000 ? Math.round(ms)+' ms' : (ms/1000).toFixed(ms<10000?2:1)+' s'; }
function trunc(s,n){ s=String(s||'').replace(/\s+/g,' ').trim(); return s.length>n ? s.slice(0,n-1)+'…' : s; }
function famOf(n){ if(n==='router')return 'routing'; if(n==='plan')return 'planning'; if(n==='rerank')return 'rerank'; if(n==='answer')return 'answer'; return 'retrieval'; }
function stepTitle(n){
  if(n==='router')return 'Router'; if(n==='plan')return 'Plan'; if(n==='retrieve')return 'Retrieve';
  if(n.startsWith('retrieve:')){ const v=n.slice(9); if(v==='hyde')return 'Retrieve · HyDE';
    if(v.startsWith('expansion_'))return 'Retrieve · expand '+v.slice(10); return 'Retrieve · '+v; }
  if(n==='pool')return 'Pool & Dedupe'; if(n==='rerank')return 'Rerank'; if(n==='answer')return 'Answer'; return n;
}
function colOf(n){ if(n==='router')return 0; if(n==='plan')return 1; if(n.startsWith('retrieve'))return 2;
  if(n==='pool')return 3; if(n==='rerank')return 4; return 5; }
function hexA(hex,a){ const n=parseInt(hex.slice(1),16); return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`; }
function dur(s){ return s.duration_ms!=null ? fmtMs(s.duration_ms) : ''; }

function stepSub(s){
  const st=s.status, d=dur(s);
  if(st==='pending') return 'pending'; if(st==='running') return '⏳ running…';
  if(st==='skipped') return 'skipped'; if(st==='cancelled') return '✕ cancelled';
  if(st==='failed') return '✕ failed'+(d?' · '+d:'');
  if(s.name.startsWith('retrieve:') && s.output && s.output.hits!=null) return s.output.hits+' hits · '+d;
  if(s.name==='pool' && s.output && s.output.sent_to_rerank!=null) return s.output.raw_hits+' → '+s.output.sent_to_rerank+' · '+d;
  if(s.name==='rerank' && s.output && s.output.kept!=null) return 'top '+s.output.kept+' · '+d;
  return '✓ '+d;
}

function stepNodeStyle(s){
  const c=FAM[famOf(s.name)], st=s.status;
  const base={borderWidth:2, shapeProperties:{borderRadius:14, borderDashes:false}, shadow:{enabled:false}, font:{color:'#e6edf3'}};
  const col=(bg,bd,hl)=>({background:bg,border:bd,highlight:{background:bg,border:hl||'#ffffff'},hover:{background:bg,border:hl||'#ffffff'}});
  if(st==='running'){ base.color=col(hexA(c,.3),c); base.borderWidth=3; base.shadow={enabled:true,color:c,size:28,x:0,y:0}; }
  else if(st==='ok'){ base.color=col(hexA(c,.18),c); }
  else if(st==='failed'){ base.color=col('#3b1219','#ef4444'); base.font={color:'#fecaca'}; base.borderWidth=3; }
  else if(st==='skipped'){ base.color=col('#161b22','#4b5563'); base.shapeProperties.borderDashes=[6,4]; base.font={color:'#6b7280'}; }
  else if(st==='cancelled'){ base.color=col('#1a1f27','#6b7280'); base.shapeProperties.borderDashes=[2,4]; base.font={color:'#6b7280'}; }
  else { base.color=col('#12171e',hexA(c,.45)); base.font={color:'#8b949e'}; base.shapeProperties.borderDashes=[3,3]; }
  return base;
}

const state = {sel: DATA.selected, showAll:false, step:null};
let network=null, nodes=null, edges=null;

function visibleTurns(){
  const t=DATA.turns;
  if(!t.length) return {list:[],hidden:0,offset:0};
  const idx=t.findIndex(x=>x.turn_id===state.sel);
  if(idx>=0 && idx<t.length-DATA.maxTurns) state.showAll=true;
  if(state.showAll||t.length<=DATA.maxTurns) return {list:t,hidden:0,offset:0};
  const off=t.length-DATA.maxTurns;
  return {list:t.slice(off),hidden:off,offset:off};
}
function ageLabel(k){ return k===0?'Current chat':(k===1?'Last chat':k+' chats ago'); }
function routeBadge(t){ return t.route==='retrieve'?'🔎':(t.route==='chat'?'💬':'…'); }
function turnTotal(t){ return t.total_ms!=null ? fmtMs(t.total_ms) : 'running…'; }

function build(){
  const {list,hidden}=visibleTurns();
  const N=DATA.turns.length, ns=[], es=[];
  const GAP=190; let x=0, prev=null, selX=null;
  if(hidden>0){ ns.push({id:'older',label:'older\n+'+hidden+' chats\n(click to expand)',x:x,y:0,shape:'box',
      color:{background:'#12171e',border:'#374151'},font:{color:'#8b949e',size:12},shapeProperties:{borderRadius:14,borderDashes:[4,4]},borderWidth:1,margin:10}); prev='older'; x+=GAP; }
  list.forEach((t,i)=>{
    const gi=(N-list.length)+i, k=N-1-gi, id='t:'+t.turn_id;
    const isCur=k===0, isSel=t.turn_id===state.sel;
    const fade=Math.max(0.38, 1-0.085*k);
    const failed=!!t.error;
    ns.push({id, x:x, y:0, shape:'box', margin:isCur?14:10, opacity:fade, borderWidth:isSel?4:(isCur?3:1.5),
      label:ageLabel(k)+'\n'+routeBadge(t)+' '+trunc(t.query,24)+'\n'+turnTotal(t),
      widthConstraint:{minimum:isCur?170:140, maximum:isCur?190:160},
      shapeProperties:{borderRadius:16},
      font:{color:isCur?'#ffffff':'#c9d1d9', size:isCur?15:13, multi:false},
      color:{background:isCur?'#0c2a3a':'#161b22', border:failed?'#ef4444':(isSel?'#e6edf3':(isCur?'#38bdf8':'#475569')),
             highlight:{background:isCur?'#0c2a3a':'#1c232c', border:'#e6edf3'}, hover:{background:'#1c232c',border:'#e6edf3'}},
      shadow: isCur ? {enabled:true,color:'#38bdf8',size:30,x:0,y:0} : {enabled:false}});
    if(prev) es.push({id:'e:'+prev+'>'+id, from:prev, to:id, arrows:{to:{enabled:true,scaleFactor:.6}}, color:{color:'#475569',opacity:fade}, width:2, smooth:false});
    if(isSel) selX=x;
    prev=id; x+=GAP;
  });

  const turn=DATA.turns.find(t=>t.turn_id===state.sel);
  if(turn){
    const steps=turn.steps||[];
    const cols={}; steps.forEach(s=>{(cols[colOf(s.name)]=cols[colOf(s.name)]||[]).push(s);});
    const maxStack=Math.max(1,...Object.values(cols).map(a=>a.length));
    const PX=215, SY=92, Y0=170, yC=Y0+(maxStack-1)*SY/2;
    const spineMid=(x-GAP)/2, x0=(selX!=null?selX:spineMid)-2.5*PX;
    const present=Object.keys(cols).map(Number).sort((a,b)=>a-b);
    const idOf=s=>'s:'+s.name, byName={};
    steps.forEach(s=>{byName[s.name]=s;});
    present.forEach(c=>{
      const arr=cols[c];
      arr.forEach((s,j)=>{
        const y=yC+(j-(arr.length-1)/2)*SY;
        ns.push(Object.assign({id:idOf(s), x:x0+c*PX, y:y, shape:'box', margin:10,
          label:stepTitle(s.name)+'\n'+stepSub(s), widthConstraint:{minimum:128,maximum:156}}, stepNodeStyle(s)));
      });
    });
    const edgeStyle=(a,b)=>{
      const tgt=b.status, c=FAM[famOf(b.name)];
      const dead=(tgt==='skipped'||tgt==='cancelled'||a.status==='skipped'||a.status==='cancelled');
      const wait=(tgt==='pending');
      return {arrows:{to:{enabled:true,scaleFactor:.7}}, width:2, dashes:(dead||wait)?[6,5]:false,
        color:{color:dead?'#374151':(tgt==='failed'?'#ef4444':(wait?hexA(c,.35):c)),opacity:dead?.8:1},
        smooth:{type:'cubicBezier',forceDirection:'horizontal',roundness:.55}};
    };
    for(let i=0;i<present.length-1;i++){
      cols[present[i]].forEach(a=>cols[present[i+1]].forEach(b=>{
        es.push(Object.assign({id:'p:'+a.name+'>'+b.name,from:idOf(a),to:idOf(b)},edgeStyle(a,b)));
      }));
    }
    if(turn.route==='chat' && byName.router && byName.answer){
      es.push({id:'p:chat-direct',from:idOf(byName.router),to:idOf(byName.answer),arrows:{to:{enabled:true,scaleFactor:.7}},width:2.5,
        color:{color:FAM.answer},smooth:{type:'curvedCW',roundness:.32}});
    }
    if(byName.router && state.sel){
      es.push({id:'p:spine>router',from:'t:'+state.sel,to:idOf(byName.router),dashes:[3,6],width:1.5,
        arrows:{to:{enabled:true,scaleFactor:.5}},color:{color:'#64748b'},smooth:{type:'cubicBezier',forceDirection:'vertical',roundness:.5}});
    }
  }
  return {ns,es};
}

function render(fit){
  const {ns,es}=build();
  if(!network){
    nodes=new vis.DataSet(ns); edges=new vis.DataSet(es);
    network=new vis.Network(document.getElementById('net'),{nodes,edges},{
      physics:false, interaction:{hover:true,dragNodes:true,zoomView:true,dragView:true,tooltipDelay:150},
      nodes:{font:{face:'Inter, system-ui, sans-serif',size:13,multi:false}, chosen:true},
      edges:{selectionWidth:0}
    });
    network.on('click',p=>{
      if(!p.nodes.length){ return; }
      const id=p.nodes[0];
      if(id==='older'){ state.showAll=true; state.step=null; render(true); renderPanel(); return; }
      if(id.startsWith('t:')){ state.sel=id.slice(2); state.step=null; render(true); renderPanel(); renderTiming(); return; }
      if(id.startsWith('s:')){ state.step=id.slice(2); renderPanel(); }
    });
    network.on('hoverNode',()=>{document.getElementById('net').style.cursor='pointer';});
    network.on('blurNode',()=>{document.getElementById('net').style.cursor='default';});
  } else {
    nodes.clear(); edges.clear(); nodes.add(ns); edges.add(es);
  }
  if(fit!==false){
    // Frame the pipeline plus the selected chat and its three predecessors;
    // the rest of the spine stays reachable by panning / zooming out.
    const ids=ns.map(n=>n.id), si=ids.indexOf('t:'+state.sel);
    const spine=ids.filter(i=>i.startsWith('t:')), k=spine.indexOf('t:'+state.sel);
    const keep=new Set(ids.filter(i=>i.startsWith('s:')));
    spine.slice(Math.max(0,k-3),k+1).forEach(i=>keep.add(i));
    const opts={animation:{duration:350,easingFunction:'easeInOutQuad'}};
    if(keep.size>1) opts.nodes=[...keep];
    network.fit(opts);
  }
}

function legend(){
  const el=document.getElementById('legend');
  let h=Object.keys(FAM).map(k=>`<span><span class="dot" style="background:${FAM[k]}"></span>${FAM_NAMES[k]}</span>`).join('');
  h+='<span><span class="dot" style="border:1px dashed #6b7280;background:transparent"></span>skipped / pending</span>';
  h+='<span><span class="dot" style="background:#ef4444"></span>failed</span>';
  if(DATA.live) h+='<span class="live">● live</span>';
  h+='<span style="margin-left:auto">click a chat to expand · click a step for details · scroll to zoom</span>';
  el.innerHTML=h;
}

function renderTiming(){
  const el=document.getElementById('timing');
  const t=DATA.turns.find(x=>x.turn_id===state.sel);
  if(!t){ el.innerHTML=''; return; }
  const rows=(t.steps||[]).filter(s=>s.duration_ms!=null);
  const mx=Math.max(1,...rows.map(s=>s.duration_ms));
  let h=`<h4>Step timing · ${esc(turnTotal(t))} total <span style="text-transform:none;letter-spacing:0">(router and plan run in parallel)</span></h4>`;
  rows.forEach(s=>{ const c=s.status==='failed'?'#ef4444':FAM[famOf(s.name)];
    h+=`<div class="trow"><span class="nm" title="${esc(stepTitle(s.name))}">${esc(stepTitle(s.name))}</span>
        <div><div class="bar" style="width:${Math.max(1,s.duration_ms/mx*100)}%;background:${c}"></div></div><span class="ms">${fmtMs(s.duration_ms)}</span></div>`; });
  if(!rows.length) h+='<div class="hint">No timings yet.</div>';
  el.innerHTML=h;
}

function chips(a){ return (a||[]).map(x=>`<span class="chip">${esc(x)}</span>`).join('')||'<span class="hint">—</span>'; }
function kv(pairs){ return '<div class="kv">'+pairs.map(([k,v])=>`<div>${esc(k)}</div><div>${v}</div>`).join('')+'</div>'; }

function renderTurnSummary(t){
  let h=`<h4>Chat</h4><div style="font-size:14px;margin-bottom:8px">${esc(t.query)}</div>`;
  h+=`<span class="badge" style="background:${t.route==='retrieve'?FAM.retrieval:FAM.routing}">${routeBadge(t)} ${esc(t.route||'…')}</span><span class="pill">${esc(turnTotal(t))}</span>`;
  if(t.model_used) h+=`<span class="pill">${esc(t.model_used)}</span>`;
  if(t.error) h+=`<div class="err">${esc(t.error)}</div>`;
  if(t.answer) h+=`<div class="sec">Answer</div><div class="box" style="font-family:inherit">${esc(trunc(t.answer,900))}</div>`;
  if((t.documents||[]).length) h+=`<div class="sec">Sources</div>`+t.documents.map(d=>`<div style="margin-bottom:4px">#${d.rank} ${esc(d.file_name)} <span class="hint">${d.score!=null?d.score.toFixed(3):''}</span></div>`).join('');
  h+='<div class="hint">Click a step in the graph to see its exact input and output.</div>';
  return h;
}

function renderStepDetail(t,s){
  const fam=famOf(s.name), c=FAM[fam], o=s.output||{};
  let h=`<h4>Step</h4><div style="margin-bottom:6px"><span class="badge" style="background:${c}">${esc(stepTitle(s.name))}</span>
    <span class="pill ${esc(s.status)}">${esc(s.status)}</span><span class="pill">${esc(dur(s)||'—')}</span></div>`;
  if(s.error) h+=`<div class="err">${esc(s.error)}</div>`;
  if(o.reason) h+=`<div class="hint">${esc(o.reason)}</div>`;
  if(s.input) h+=`<div class="sec">Input</div><div class="box">${esc(s.input)}</div>`;
  const n=s.name;
  if(n==='router'){
    h+=`<div class="sec">Output</div>`+kv([['decision','<b>'+esc(o.decision)+'</b>'],['raw reply',esc(o.raw)]]);
  } else if(n==='plan'){
    if(o.rewrite!=null) h+=`<div class="sec">Rewrite</div><div class="box">${esc(o.rewrite)||'—'}</div>`;
    h+=`<div class="sec">Expansions (${(o.expansions||[]).length})</div>`+((o.expansions||[]).length?'<ol style="margin:0;padding-left:20px">'+o.expansions.map(e=>`<li>${esc(e)}</li>`).join('')+'</ol>':'<span class="hint">—</span>');
    h+=`<div class="sec">HyDE paragraph</div><div class="box">${esc(o.hyde)||'—'}</div>`;
    h+=`<div class="sec">Variants sent to retrieval</div>`+chips(o.variants);
    if(o.raw) h+=`<div class="sec">Raw planner output</div><pre>${esc(o.raw)}</pre>`;
  } else if(n.startsWith('retrieve:')){
    h+=`<div class="sec">Output</div>`+kv([['retriever',esc(o.retriever)],['hits',esc(o.hits)]]);
    h+=`<div class="sec">Files</div>`+chips(o.files);
    h+=`<div class="sec">Top hits</div>`+(o.top||[]).map(d=>`<div class="box" style="margin-bottom:6px"><b>${esc(d.file_name)}</b>\n${esc(d.snippet)}</div>`).join('');
  } else if(n==='pool'){
    h+=`<div class="sec">Output</div>`+kv([['raw hits',esc(o.raw_hits)],['blank (empty-text) hits',o.blank_hits?'<b style="color:#f85149">'+esc(o.blank_hits)+'</b>':'0'],['after dedupe',esc(o.after_dedupe)],['sent to rerank',esc(o.sent_to_rerank)+' (cap '+esc(o.cap)+')'],['dropped by cap',esc(o.dropped_by_cap)],['merge order',o.interleaved?'round-robin across variants':'concatenated']]);
    const ph=o.per_variant_hits||{}, pp=o.per_variant_in_pool||{};
    h+=`<div class="sec">Per variant: hits → chunks in pool</div><table><tr><th>variant</th><th>hits</th><th>in pool</th></tr>`+Object.keys(ph).map(k=>`<tr><td>${esc(k)}</td><td>${ph[k]}</td><td>${pp[k]||0}</td></tr>`).join('')+'</table>';
  } else if(n==='rerank'){
    h+=`<div class="sec">Output</div>`+kv([['model',esc(o.model)],['input → kept',esc(o.input_count)+' → '+esc(o.kept)],['fallback',o.fallback?'<b style="color:#f85149">yes (reranker failed)</b>':'no']]);
    const docs=o.documents||[]; const mx=Math.max(0.0001,...docs.map(d=>Math.abs(d.score||0)));
    h+=`<div class="sec">Reranked chunks</div>`+docs.map(d=>`<div class="box" style="margin-bottom:6px"><b>#${d.rank} ${esc(d.file_name)}</b> · score ${d.score!=null?d.score.toFixed(3):'n/a'}
       ${d.score!=null?`<div class="miniBar" style="width:${Math.max(3,Math.abs(d.score)/mx*100)}%"></div>`:''}
       <div style="margin:4px 0">${chips(d.found_by)}</div>${esc(d.snippet)}</div>`).join('')||'<span class="hint">no documents</span>';
  } else if(n==='answer'){
    h+=`<div class="sec">Output</div>`+kv([['model',esc(o.model)+(o.fallback_used?' <span class="pill failed">fallback</span>':'')],['context chunks',esc(o.context_chunks)],['characters',esc(o.chars)]]);
    if(t.answer) h+=`<div class="sec">Answer</div><div class="box" style="font-family:inherit">${esc(t.answer)}</div>`;
    if(t.reasoning) h+=`<details style="margin-top:8px"><summary>Reasoning</summary><pre>${esc(t.reasoning)}</pre></details>`;
  } else if(Object.keys(o).length){
    h+=`<div class="sec">Output</div><pre>${esc(JSON.stringify(o,null,2))}</pre>`;
  }
  return h;
}

function renderPanel(){
  const el=document.getElementById('panel');
  const t=DATA.turns.find(x=>x.turn_id===state.sel);
  if(!t){ el.innerHTML='<h4>Pipeline</h4><div class="hint">No chats yet. Ask something in the Chat tab and its pipeline will appear here.</div>'; return; }
  const s=state.step && (t.steps||[]).find(x=>x.name===state.step);
  el.innerHTML = s ? renderStepDetail(t,s)+`<div class="hint"><a href="#" id="back" style="color:#58a6ff">← back to chat summary</a></div>` : renderTurnSummary(t);
  const b=document.getElementById('back'); if(b) b.onclick=e=>{e.preventDefault(); state.step=null; renderPanel();};
}

(function init(){
  legend();
  if(typeof vis==='undefined'){
    document.getElementById('net').innerHTML='<div id="nolib">Could not load vis-network from unpkg.com (offline?). Allow that host or vendor the script into graph_view.py.</div>';
    renderPanel(); renderTiming(); return;
  }
  if(!state.sel && DATA.turns.length) state.sel=DATA.turns[DATA.turns.length-1].turn_id;
  render(true); renderPanel(); renderTiming();
})();
</script></body></html>
"""


def graph_html(
    turns: list[dict[str, Any]],
    selected_id: str | None = None,
    live: bool = False,
    max_turns: int = 8,
) -> str:
    """Return the page for `components.html`. `turns` are TurnTrace.to_dict() dicts, oldest first."""
    if turns and selected_id not in {t["turn_id"] for t in turns}:
        selected_id = turns[-1]["turn_id"]
    payload = {"turns": turns, "selected": selected_id, "live": live, "maxTurns": max_turns}
    data = json.dumps(payload, default=str).replace("</", "<\\/")
    return _TEMPLATE.replace("__VIS_URL__", VIS_URL).replace("__DATA__", data)
