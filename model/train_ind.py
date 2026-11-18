import argparse
import copy
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

from lib.utils import StandardScaler, vrange, seed_everything, set_cpu_num
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from STAEformer_var import STAEformerVar
from descriptors import sensor_descriptors, normalise_descriptors
from train_var import VARIANTS, count_params, metrics_block, predict


def make_split(n_nodes, split_seed, frac=0.2):
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(n_nodes)
    n_new = int(round(frac * n_nodes))
    return np.sort(perm[n_new:]), np.sort(perm[:n_new])  # seen, new


def load_windows(data_dir):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)[..., :3]
    index = np.load(os.path.join(data_dir, "index.npz"))
    out = {}
    for s in ("train", "val", "test"):
        i = index[s]
        xi = vrange(i[:, 0], i[:, 1])
        yi = vrange(i[:, 1], i[:, 2])
        out[s] = (data[xi], data[yi][..., :1])
    return out


@torch.no_grad()
def predict_array(model, x, scaler, device, bs=16):
    model.eval()
    outs = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(device)
        outs.append(scaler.inverse_transform(model(xb)).cpu().numpy())
    return np.concatenate(outs).squeeze(-1)  # (S, 12, N)


def make_model(vkey, model_args, n_nodes, args, device):
    variant, overrides = VARIANTS[vkey]
    margs = dict(model_args, **overrides)
    margs["num_nodes"] = n_nodes
    return STAEformerVar(
        variant=variant, time_dim=args.time_dim, hidden=args.desc_hidden, **margs
    ).to(device)


def eval_context(vkey, sd, model_args, mapping, z_eval, x, scaler, args, device):
    """Evaluate trained weights on a sensor set of a different size.

    mapping: for original/factorised, row of the trained embedding to use for each
    evaluation sensor (own row for seen sensors, nearest seen sensor for new ones).
    """
    n_eval = len(mapping)
    model = make_model(vkey, model_args, n_eval, args, device)
    sd = dict(sd)
    if vkey in ("original", "factorised"):
        key = "adaptive_embedding" if vkey == "original" else "node_table"
        mp = torch.as_tensor(mapping, dtype=torch.long, device=sd[key].device)
        sd[key] = sd[key][:, mp] if vkey == "original" else sd[key][mp]
    model.load_state_dict(sd)
    if vkey == "generated":
        model.set_descriptors(torch.tensor(z_eval, dtype=torch.float32))
    return predict_array(model, x, scaler, device)


def run_train(args):
    dataset = args.dataset.upper()
    vkey = args.variant
    tag = f"{dataset}_{vkey}_split{args.split_seed}"
    out_dir = os.path.join(ROOT, "results", "ind_smoke" if args.smoke else "ind")
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
    if args.smoke:
        cfg["max_epochs"] = 1
    model_args = cfg["model_args"]

    W = load_windows(data_path)
    n_all = W["train"][0].shape[2]
    seen_idx, new_idx = make_split(n_all, args.split_seed)
    print(f"{tag}: {len(seen_idx)} seen sensors, {len(new_idx)} held out")

    # scaler fitted on the seen sensors only
    xtr = W["train"][0]
    scaler = StandardScaler(mean=xtr[:, :, seen_idx, 0].mean(),
                            std=xtr[:, :, seen_idx, 0].std())
    raw_test_flow = W["test"][0][..., 0].copy()
    for s in W:
        W[s][0][..., 0] = scaler.transform(W[s][0][..., 0])
    xte, yte = W["test"]

    base = {"tag": tag, "dataset": dataset, "variant": vkey,
            "split_seed": args.split_seed, "seed": args.seed, "smoke": args.smoke,
            "n_seen": int(len(seen_idx)), "n_new": int(len(new_idx)),
            "new_idx": new_idx.tolist()}

    # Historical Inertia: the forecast is a copy of the input window, no training
    if vkey == "hi":
        res = dict(base, new_in_all=metrics_block(yte[:, :, new_idx, 0],
                                                  raw_test_flow[:, :, new_idx]))
        json.dump(res, open(res_path, "w"))
        print(tag, res["new_in_all"])
        return

    z = normalise_descriptors(sensor_descriptors(data_path), seen_idx)

    def loader(x, y, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x[:, :, seen_idx])),
            torch.from_numpy(np.ascontiguousarray(y[:, :, seen_idx])))
        return torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    train_loader = loader(*W["train"], True)
    val_loader = loader(*W["val"], False)

    model = make_model(vkey, model_args, len(seen_idx), args, device)
    if vkey == "generated":
        model.set_descriptors(torch.tensor(z[seen_idx], dtype=torch.float32))
    n_total, n_emb = count_params(model)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 0),
                                 eps=cfg.get("eps", 1e-8))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["milestones"], gamma=cfg.get("lr_decay_rate", 0.1))

    wait, best, best_epoch, best_state = 0, np.inf, 0, None
    times = []
    t_start = time.time()
    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        for x, y in train_loader:
            out = scaler.inverse_transform(model(x.to(device)))
            loss = criterion(out, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        vl = []
        with torch.no_grad():
            for x, y in val_loader:
                out = scaler.inverse_transform(model(x.to(device)))
                vl.append(criterion(out, y.to(device)).item())
        val_loss = float(np.mean(vl))
        times.append(time.time() - t0)
        print(f"epoch {epoch + 1} val {val_loss:.4f} ({times[-1]:.0f}s)", flush=True)
        if val_loss < best:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["early_stop"]:
                break
    train_seconds = time.time() - t_start

    # nearest seen sensor (by descriptor) for every new sensor
    d = ((z[new_idx][:, None, :] - z[seen_idx][None, :, :]) ** 2).sum(-1)
    nn_rank = d.argmin(1)
    seen_rank = {int(s): r for r, s in enumerate(seen_idx)}
    nn_of = {int(s): int(r) for s, r in zip(new_idx, nn_rank)}
    map_all = np.array([seen_rank.get(j, nn_of.get(j)) for j in range(n_all)])

    # (1) all sensors in the attention context, score the held-out ones
    pred_all = eval_context(vkey, best_state, model_args, map_all, z, xte, scaler, args, device)
    new_in_all = metrics_block(yte[:, :, new_idx, 0], pred_all[:, :, new_idx])
    seen_in_all = metrics_block(yte[:, :, seen_idx, 0], pred_all[:, :, seen_idx])
    # (2) only the held-out sensors in the context
    pred_new = eval_context(vkey, best_state, model_args, nn_rank, z[new_idx],
                            xte[:, :, new_idx], scaler, args, device)
    new_only = metrics_block(yte[:, :, new_idx, 0], pred_new)

    res = dict(base, params_total=n_total, params_embedding=n_emb,
               epochs_run=epoch + 1, best_epoch=best_epoch + 1,
               sec_per_epoch=float(np.mean(times)), train_seconds=train_seconds,
               new_in_all=new_in_all, new_only=new_only, seen_in_all=seen_in_all)
    json.dump(res, open(res_path, "w"))
    print(tag, "new_in_all", new_in_all, "| new_only", new_only, "| seen_in_all", seen_in_all)


def run_ref(args):
    """Transductive models (trained on all sensors) scored on the same held-out sensors."""
    dataset = args.dataset.upper()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(ROOT, "data", dataset)
    with open(os.path.join(HERE, "STAEformer.yaml")) as f:
        cfg = yaml.safe_load(f)[dataset]
    _, _, test_loader, scaler = get_dataloaders_from_index_data(
        data_path, tod=True, dow=True, batch_size=cfg["batch_size"])
    n_all = cfg["num_nodes"]
    z = normalise_descriptors(sensor_descriptors(data_path), np.arange(n_all))
    out_dir = os.path.join(ROOT, "results", "ind")
    os.makedirs(out_dir, exist_ok=True)
    for vkey in ("original", "factorised", "generated", "noadp"):
        for seed in (0, 1, 2):
            ckpt = os.path.join(ROOT, "saved_models", f"{dataset}_{vkey}_s{seed}.pt")
            if not os.path.exists(ckpt):
                print("missing", ckpt)
                continue
            model = make_model(vkey, cfg["model_args"], n_all, args, device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            if vkey == "generated":
                model.set_descriptors(torch.tensor(z, dtype=torch.float32))
            y, p = predict(model, test_loader, scaler, device)
            res = {"dataset": dataset, "variant": vkey, "seed": seed, "splits": {}}
            for k in (0, 1, 2):
                _, new_idx = make_split(n_all, k)
                res["splits"][str(k)] = metrics_block(y[..., new_idx], p[..., new_idx])
            json.dump(res, open(os.path.join(out_dir, f"ref_{dataset}_{vkey}_s{seed}.json"), "w"))
            print(dataset, vkey, seed, res["splits"]["0"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "ref"], default="train")
    ap.add_argument("-d", "--dataset", default="PEMS08")
    ap.add_argument("-g", "--gpu_num", type=int, default=0)
    ap.add_argument("--variant", choices=["original", "noadp", "factorised", "generated", "hi"],
                    default="generated")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--time_dim", type=int, default=16)
    ap.add_argument("--desc_hidden", type=int, default=128)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = args.split_seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    run_train(args) if args.mode == "train" else run_ref(args)
