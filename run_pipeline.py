import subprocess
import sys

def run_step(script_name):
    print(f"\n🚀 Running {script_name}...\n")
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f"❌ Error in {script_name}")
        print(result.stderr)
        sys.exit(1)

    print(f"✅ Finished {script_name}")

# =========================
# PIPELINE ORDER
# =========================

pipeline = [
    "NewdDataFetch.py",
    "Pravaayu_TransformerScript.py",
    "Pruning.py",
    "Apriori.py"
]

for step in pipeline:
    run_step(step)

print("\n🎯 FULL PIPELINE COMPLETED SUCCESSFULLY")