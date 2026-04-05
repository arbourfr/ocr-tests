"""Test the vLLM-based GLM-OCR endpoint."""
from __future__ import annotations
import argparse, base64, json, os, sys, time
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("RUNPOD_API_KEY")
EP = os.getenv("RUNPOD_VLLM_ENDPOINT_ID")
if not (API_KEY and EP):
    sys.exit("Need RUNPOD_API_KEY and RUNPOD_VLLM_ENDPOINT_ID in .env")

H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(ROOT / "samples/acte_10175367.pdf"))
    ap.add_argument("--pages", nargs="+", type=int)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()
    payload = {"input": {"pdf_base64": pdf_b64, "dpi": args.dpi, "max_new_tokens": args.max_new_tokens}}
    if args.pages:
        payload["input"]["pages"] = args.pages

    print(f"[test] pdf={pdf_path.name}  size={pdf_path.stat().st_size:,}")
    r = requests.post(f"https://api.runpod.ai/v2/{EP}/run", headers=H, json=payload, timeout=60)
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"[submit] job={job_id}")

    t0 = time.time()
    last = None
    while True:
        s = requests.get(f"https://api.runpod.ai/v2/{EP}/status/{job_id}", headers=H, timeout=30)
        s.raise_for_status()
        data = s.json()
        st = data.get("status")
        if st != last:
            print(f"[poll] status={st}  ({time.time()-t0:.1f}s)")
            last = st
        if st in ("COMPLETED","FAILED","CANCELLED","TIMED_OUT"):
            break
        if time.time() - t0 > args.timeout:
            print(f"[poll] client timeout {args.timeout}s; job may still be running server-side")
            print(f"       server job_id={job_id}")
            sys.exit(1)
        time.sleep(5)

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    raw = out_dir / f"{pdf_path.stem}.vllm.json"
    raw.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[out] {raw}")

    if data.get("status") != "COMPLETED":
        print(json.dumps(data, indent=2)[:1500])
        sys.exit(1)
    o = data["output"]
    if "error" in o:
        sys.exit(f"worker error: {o['error']}")

    md = out_dir / f"{pdf_path.stem}.vllm.md"
    with md.open("w") as f:
        f.write(f"# OCR — {pdf_path.name} (GLM-OCR / vLLM)\n\n")
        f.write(f"Elapsed: {o['elapsed_s']}s for {o['num_pages']} pages = {o['elapsed_s']/o['num_pages']:.2f}s/page  |  concurrency={o.get('concurrency')}\n\n---\n")
        for p in o["pages"]:
            f.write(f"\n## Page {p['page']}\n\n{p['text']}\n")
    print(f"[out] {md}")
    print(f"[done] {o['elapsed_s']}s / {o['num_pages']} pages = {o['elapsed_s']/o['num_pages']:.2f}s/page")

if __name__ == "__main__":
    main()
