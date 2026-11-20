import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.append(ROOT)

from lib.data_prepare import get_dataloaders_from_index_data
from lib.metrics import RMSE_MAE_MAPE
from STAEformer_var import STAEformerVar
from descriptors import sensor_descriptors, normalise_descriptors
from train_var import VARIANTS

HORIZONS = (3, 6, 12)


@torch.no_grad()
def run(model, loader, scaler, device, mode, seed):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        if mode != "none":
            B, T, N, C = x.shape
            perm = torch.argsort(torch.rand(B, T, generator=gen), dim=1).to(device)
            idx = perm[:, :, None, None]
            if mode == "frames":
                x = torch.gather(x, 1, idx.expand(-1, -1, N, C))
            else:
                flow = torch.gather(x[..., :1], 1, idx.expand(-1, -1, N, 1))
                x = torch.cat([flow, x[..., 1:]], dim=-1)
        ps.append(scaler.inverse_transform(model(x)).cpu().numpy())
        ys.append(y.numpy())
    return np.vstack(ys).squeeze(), np.vstack(ps).squeeze()


def mae_set(y, p):
    out = {"all": float(RMSE_MAE_MAPE(y, p)[1])}
    for h in HORIZONS:
        out[f"h{h}"] = float(RMSE_MAE_MAPE(y[:, h - 1, :], p[:, h - 1, :])[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gpu_num", type=int, default=0)
    ap.add_argument("--mode", choices=["frames", "flow"], default="frames")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(ROOT, "results", "shuffle")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for ds in ("PEMS08", "PEMS04"):
        with open(os.path.join(HERE, "STAEformer.yaml")) as f:
            cfg = yaml.safe_load(f)[ds]
        data_path = os.path.join(ROOT, "data", ds)
        _, _, test_loader, scaler = get_dataloaders_from_index_data(
            data_path, tod=True, dow=True, batch_size=cfg["batch_size"])
        n_all = cfg["num_nodes"]
        z = normalise_descriptors(sensor_descriptors(data_path), np.arange(n_all))
        for vkey in ("original", "spatial", "factorised", "generated"):
            for seed in (0, 1, 2):
                ckpt = os.path.join(ROOT, "saved_models", f"{ds}_{vkey}_s{seed}.pt")
                if not os.path.exists(ckpt):
                    print("missing", ckpt)
                    continue
                variant, overrides = VARIANTS[vkey]
                model = STAEformerVar(variant=variant, **dict(cfg["model_args"], **overrides))
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model.to(device)
                if vkey == "generated":
                    model.set_descriptors(torch.tensor(z, dtype=torch.float32))
                normal = mae_set(*run(model, test_loader, scaler, device, "none", 0))
                shuf = mae_set(*run(model, test_loader, scaler, device, args.mode, 123 + seed))
                inc = {k: 100 * (shuf[k] - normal[k]) / normal[k] for k in normal}
                res = {"dataset": ds, "variant": vkey, "seed": seed, "mode": args.mode,
                       "normal": normal, "shuffled": shuf, "increase_pct": inc}
                json.dump(res, open(os.path.join(
                    out_dir, f"{ds}_{vkey}_s{seed}_{args.mode}.json"), "w"))
                summary.setdefault((ds, vkey), []).append(inc)
                print(ds, vkey, seed, {k: round(v, 1) for k, v in inc.items()}, flush=True)

    print(f"\n=== MAE increase (%) after shuffling ({args.mode}), mean ± std over seeds ===")
    print(f"{'data':7s}{'variant':11s}{'all':>14s}{'h3':>14s}{'h6':>14s}{'h12':>14s}")
    for (ds, v), incs in summary.items():
        cells = []
        for k in ("all", "h3", "h6", "h12"):
            a = np.array([i[k] for i in incs])
            cells.append(f"{a.mean():.1f} ± {(a.std(ddof=1) if len(a) > 1 else 0):.1f}")
        print(f"{ds:7s}{v:11s}" + "".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
