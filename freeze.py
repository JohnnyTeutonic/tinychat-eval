"""Offline manifest freeze/check; no model, credential or provider access."""
import argparse,hashlib,json
from pathlib import Path
FILES=("evaluate.py","integrations.py","suite.json","suite_release_v002.json","release_gate_v002.json","training_lr3e5_v002.json","training_lr1e4_v002.json","V002_RELEASE.md","freeze.py","training_data_v002.json")
def manifest(root):
 return {"version":"v0.0.2-draft1","files":{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in FILES},"status":"FROZEN_PROPOSAL_NOT_LAUNCH_APPROVAL"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--out");p.add_argument("--check");a=p.parse_args();root=Path(__file__).parent;m=manifest(root)
 if bool(a.out)==bool(a.check):p.error("exactly one of --out/--check")
 if a.check:
  if json.loads(Path(a.check).read_text())!=m:raise SystemExit("manifest drift")
  print("Frozen evaluation manifest verified")
 else:
  with Path(a.out).open("x",encoding="utf-8") as f:json.dump(m,f,indent=2)
if __name__=="__main__":main()
