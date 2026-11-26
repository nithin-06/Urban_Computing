import subprocess
import sys
import time

gpus = [int(g) for g in (sys.argv[1] if len(sys.argv) > 1 else "0,1").split(",")]
jobs = [(d, s) for s in (0, 1, 2) for d in ("PEMS08", "PEMS04")]
running, queue, failed = {}, list(jobs), []
while queue or running:
    for g in gpus:
        if g in running and running[g][0].poll() is not None:
            proc, job = running.pop(g)
            if proc.returncode != 0:
                failed.append(job)
            print("finished" if proc.returncode == 0 else "FAILED", job, flush=True)
        if g not in running and queue:
            d, s = queue.pop(0)
            cmd = [sys.executable, "train_ind.py", "-d", d, "-g", str(g), "--variant",
                   "generated", "--split_seed", str(s), "--desc_days", "28"]
            out = open(f"../logs/queue_d28_{d}_split{s}.out", "w")
            running[g] = (subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), (d, s))
            print("started", (d, s), "on gpu", g, flush=True)
    time.sleep(20)
print("done; failed:", failed)
