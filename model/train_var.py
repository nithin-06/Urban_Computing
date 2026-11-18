import argparse
import copy
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.append(ROOT)

from lib.utils import MaskedMAELoss, print_log, seed_everything, set_cpu_num
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from STAEformer_var import STAEformerVar
from descriptors import sensor_descriptors, normalise_descriptors

# name -> (embedding variant, overrides of model_args)
VARIANTS = {
    "original": ("full", {}),
    "noadp": ("full", {"adaptive_embedding_dim": 0}),
    "spatial": ("full", {"adaptive_embedding_dim": 0, "spatial_embedding_dim": 80}),
    "factorised": ("factorised", {}),
    "generated": ("generated", {}),
}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    # parameters registered directly on the model are the embedding tables
    emb = sum(p.numel() for p in model._parameters.values() if p is not None)
    if hasattr(model, "node_mlp"):
        emb += sum(p.numel() for p in model.node_mlp.parameters())
    return total, emb


@torch.no_grad()
def eval_loss(model, loader, scaler, criterion, device):
    model.eval()
    losses = []
    for x, y in loader:
        out = scaler.inverse_transform(model(x.to(device)))
        losses.append(criterion(out, y.to(device)).item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, scaler, device):
    model.eval()
    ys, outs = [], []
    for x, y in loader:
        out = scaler.inverse_transform(model(x.to(device)))
        outs.append(out.cpu().numpy())
        ys.append(y.numpy())
    return np.vstack(ys).squeeze(), np.vstack(outs).squeeze()  # (S, 12, N)


def metrics_block(y, p):
    rmse, mae, mape = RMSE_MAE_MAPE(y, p)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def per_sensor_stats(y, p):
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    err = np.abs(y - p)
    mask = y != 0
    ape = np.where(mask, err / np.where(mask, np.abs(y), 1.0), 0.0)
    return {
        "mae": err.mean(axis=(0, 1)).tolist(),
        "mse": ((y - p) ** 2).mean(axis=(0, 1)).tolist(),
        "ape_sum": ape.sum(axis=(0, 1)).tolist(),
        "ape_cnt": mask.sum(axis=(0, 1)).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="PEMS08")
    ap.add_argument("-g", "--gpu_num", type=int, default=0)
    ap.add_argument("--variant", choices=list(VARIANTS), default="original")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_dim", type=int, default=16)
    ap.add_argument("--desc_hidden", type=int, default=128)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, results/smoke")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    dataset = args.dataset.upper()
    tag = args.tag or f"{dataset}_{args.variant}_s{args.seed}"
    out_dir = os.path.join(ROOT, "results", "smoke" if args.smoke else "runs")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, f"{tag}.json")
    if os.path.exists(res_path) and not args.force:
        print("already done, skipping:", res_path)
        return

    seed_everything(args.seed)
    set_cpu_num(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(ROOT, "data", dataset)

    with open(os.path.join(HERE, "STAEformer.yaml")) as f:
        cfg = yaml.safe_load(f)[dataset]
    variant, overrides = VARIANTS[args.variant]
    model_args = dict(cfg["model_args"], **overrides)
    if args.smoke:
        cfg["max_epochs"] = 1

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "saved_models"), exist_ok=True)
    log = open(os.path.join(ROOT, "logs", f"{tag}.log"), "w")
    save = os.path.join(ROOT, "saved_models", f"{tag}.pt")
    print_log(tag, json.dumps(cfg, indent=2, default=str), log=log)

    train_loader, val_loader, test_loader, scaler = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )

    model = STAEformerVar(
        variant=variant, time_dim=args.time_dim, hidden=args.desc_hidden, **model_args
    ).to(device)
    if args.variant == "generated":
        z = sensor_descriptors(data_path)
        z = normalise_descriptors(z, np.arange(z.shape[0]))
        model.set_descriptors(torch.tensor(z, dtype=torch.float32))
    n_total, n_emb = count_params(model)
    print_log(f"params total={n_total:,} embedding={n_emb:,}", log=log)

    criterion = MaskedMAELoss() if dataset in ("METRLA", "PEMSBAY") else nn.HuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=cfg.get("lr_decay_rate", 0.1)
    )
    clip_grad = cfg.get("clip_grad")

    # ------------------------------- training ------------------------------- #
    wait, best, best_epoch, best_state = 0, np.inf, 0, None
    epoch_times = []
    t_start = time.time()
    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        losses = []
        for x, y in train_loader:
            out = scaler.inverse_transform(model(x.to(device)))
            loss = criterion(out, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val_loss = eval_loss(model, val_loader, scaler, criterion, device)
        epoch_times.append(time.time() - t0)
        print_log(datetime.datetime.now(), "Epoch", epoch + 1,
                  "\tTrain Loss = %.5f" % np.mean(losses),
                  "Val Loss = %.5f" % val_loss, log=log)
        if val_loss < best:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["early_stop"]:
                break
    train_seconds = time.time() - t_start

    model.load_state_dict(best_state)
    torch.save(best_state, save)

    # ------------------------------- evaluation ------------------------------ #
    yv, pv = predict(model, val_loader, scaler, device)
    t0 = time.time()
    yt, pt = predict(model, test_loader, scaler, device)
    infer_seconds = time.time() - t0

    res = {
        "tag": tag, "dataset": dataset, "variant": args.variant, "seed": args.seed,
        "time_dim": args.time_dim, "smoke": args.smoke,
        "params_total": n_total, "params_embedding": n_emb,
        "epochs_run": epoch + 1, "best_epoch": best_epoch + 1,
        "sec_per_epoch": float(np.mean(epoch_times)),
        "train_seconds": train_seconds, "infer_seconds": infer_seconds,
        "val": metrics_block(yv, pv),
        "test": metrics_block(yt, pt),
        "test_steps": [metrics_block(yt[:, i, :], pt[:, i, :]) for i in range(yt.shape[1])],
        "test_per_sensor": per_sensor_stats(yt, pt),
    }
    with open(res_path, "w") as f:
        json.dump(res, f)
    print_log("Test:", res["test"], log=log)
    log.close()
    print("saved", res_path, "| test", res["test"])


if __name__ == "__main__":
    main()
