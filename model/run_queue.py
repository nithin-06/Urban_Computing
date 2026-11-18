import subprocess
import sys
import time

gpus = [int(g) for g in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]
variants = ["original", "noadp", "spatial", "factorised", "generated"]
jobs = [(d, v, s) for s in (0, 1, 2) for d in ("PEMS08", "PEMS04") for v in variants]

running = {}  # gpu -> (Popen, job)
queue = list(jobs)
failed = []
while queue or running:
    for g in gpus:
        if g in running and running[g][0].poll() is not None:
            proc, job = running.pop(g)
            if proc.returncode != 0:
                failed.append(job)
            print("finished" if proc.returncode == 0 else "FAILED", job, flush=True)
        if g not in running and queue:
            d, v, s = queue.pop(0)
            cmd = [sys.executable, "train_var.py", "-d", d, "-g", str(g),
                   "--variant", v, "--seed", str(s)]
            out = open(f"../logs/queue_{d}_{v}_s{s}.out", "w")
            running[g] = (subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), (d, v, s))
            print("started", (d, v, s), "on gpu", g, flush=True)
    time.sleep(20)
print("done; failed:", failed)
