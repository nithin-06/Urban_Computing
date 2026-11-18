import glob, json

print(f"{'data':7s}{'variant':11s}{'seed':>5s}{'ep':>5s}{'best':>5s}{'min':>7s}"
      f"{'s/ep':>7s}{'MAE':>8s}{'RMSE':>8s}{'MAPE':>7s}{'params':>11s}")
for f in sorted(glob.glob("../results/runs/*.json")):
    r = json.load(open(f))
    t = r["test"]
    print(f"{r['dataset']:7s}{r['variant']:11s}{r['seed']:5d}{r['epochs_run']:5d}"
          f"{r['best_epoch']:5d}{r['train_seconds']/60:7.1f}{r['sec_per_epoch']:7.1f}"
          f"{t['mae']:8.3f}{t['rmse']:8.3f}{t['mape']:7.2f}{r['params_total']:11,d}")
