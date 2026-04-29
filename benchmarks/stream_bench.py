#!/usr/bin/env python3
import os
import json, time, urllib.request, threading, queue
from pathlib import Path
HOST=os.environ.get("BASE_URL", "http://127.0.0.1:18141").rstrip("/").removesuffix("/v1")
KEY=os.environ.get("API_KEY", "change-me")
MODEL="Qwen3.6-27B-AWQ4"
OUT=Path(os.environ.get("BENCH_OUT", "stream_tuning_results.local.jsonl"))
unit="""\nFile app.py: validate JSON input, call local OpenAI-compatible model endpoint, retry transient errors, return structured JSON. Requirements: no secrets in logs, timeouts, clear errors, tests for malformed input.\n"""
TESTS=[
 ("short_stream", "Write a concise Python function that adds two numbers. Output code only.", 256),
 ("medium_stream", ("You are a precise coding agent. Context follows.\n"+unit*55+"\nNow write the implementation plan and code."), 512),
 ("long_stream", ("You are a precise coding agent. Context follows.\n"+unit*620+"\nNow write the implementation plan and code."), 512),
]

def sse(path, body, timeout=1800):
    req=urllib.request.Request(HOST+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json","Accept":"text/event-stream","Authorization":"Bearer "+KEY}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ev=None; data=[]
        for raw in resp:
            line=raw.decode("utf-8","replace").rstrip("\r\n")
            if line=="":
                if ev or data:
                    payload="\n".join(data)
                    if payload=="[DONE]": return
                    try: obj=json.loads(payload) if payload else {}
                    except Exception: obj={"_raw":payload}
                    yield ev or obj.get("type"), obj
                ev=None; data=[]; continue
            if line.startswith("event:"): ev=line[6:].strip()
            elif line.startswith("data:"): data.append(line[5:].lstrip())

def one(name,prompt,max_tokens):
    body={"model":MODEL,"input":prompt,"max_output_tokens":max_tokens,"temperature":0,"stream":True,"chat_template_kwargs":{"enable_thinking":False},"parallel_tool_calls":True}
    t0=time.perf_counter(); first=None; out_chars=0; reasoning_chars=0; events={}; completed={}; output_preview=[]
    for ev,obj in sse("/v1/responses", body):
        now=time.perf_counter()
        if first is None: first=now
        events[ev]=events.get(ev,0)+1
        if ev=="response.output_text.delta":
            d=obj.get("delta",""); out_chars+=len(d); output_preview.append(d)
        elif ev=="response.reasoning_text.delta":
            d=obj.get("delta",""); reasoning_chars+=len(d)
        elif ev=="response.completed":
            completed=obj.get("response",obj)
    t1=time.perf_counter(); usage=completed.get("usage",{}) if isinstance(completed,dict) else {}
    out_tok=usage.get("output_tokens") or usage.get("completion_tokens") or 0
    in_tok=usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    total=usage.get("total_tokens") or (in_tok+out_tok)
    row={"name":name,"mode":"responses_stream_think_off","input_tokens":in_tok,"output_tokens":out_tok,"total_tokens":total,"max_output_tokens":max_tokens,"elapsed_s":round(t1-t0,3),"ttft_s":round((first-t0),3) if first else None,"output_tps":round(out_tok/(t1-t0),3) if out_tok else None,"visible_chars":out_chars,"reasoning_chars":reasoning_chars,"events":events,"sample":"".join(output_preview)[:300]}
    print(json.dumps(row,ensure_ascii=False), flush=True)
    with OUT.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return row

def worker(reqs, resq):
    for x in reqs:
        try: resq.put(one(*x))
        except Exception as e: resq.put({"error":repr(e),"name":x[0]})

def parallel_medium(n=3):
    prompt="You are a concise coding agent.\n"+unit*35+"\nReturn a robust Python helper and short notes."
    reqs=[(f"parallel_medium_{i+1}",prompt,512) for i in range(n)]
    qs=[]; threads=[]; t0=time.perf_counter(); res=[]
    for r in reqs:
        q=queue.Queue(); th=threading.Thread(target=lambda rr=r,qq=q: qq.put(one(*rr)), daemon=True); th.start(); qs.append(q); threads.append(th)
    for th,q in zip(threads,qs):
        th.join(); res.append(q.get())
    elapsed=time.perf_counter()-t0
    toks=sum((r.get("output_tokens") or 0) for r in res)
    row={"name":f"parallel_medium_{n}_aggregate","mode":"responses_stream_think_off","requests":n,"elapsed_s":round(elapsed,3),"output_tokens":toks,"aggregate_output_tps":round(toks/elapsed,3) if elapsed else None,"per_request_tps":[r.get("output_tps") for r in res],"rows":res}
    print(json.dumps(row,ensure_ascii=False), flush=True)
    with OUT.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")

if __name__=="__main__":
    for t in TESTS: one(*t)
    parallel_medium(3)
