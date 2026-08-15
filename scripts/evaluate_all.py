from __future__ import annotations
import subprocess,sys
for module in ("scripts.evaluate_retrieval","scripts.evaluate_rag","scripts.evaluate_abstention"):
    print("\n"+"="*80); print("RUNNING",module); print("="*80)
    rc=subprocess.run([sys.executable,"-m",module]).returncode
    if rc: raise SystemExit(rc)
print("\nALL EVALUATIONS COMPLETED")
