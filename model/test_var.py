import torch
from STAEformer_var import STAEformerVar

ARGS = dict(
    num_nodes=170, in_steps=12, out_steps=12, steps_per_day=288,
    input_dim=3, output_dim=1, input_embedding_dim=24, tod_embedding_dim=24,
    dow_embedding_dim=24, spatial_embedding_dim=0, adaptive_embedding_dim=80,
    feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1,
)


def fake_batch(b, n):
    x = torch.rand(b, 12, n, 3)
    x[..., 2] = torch.randint(0, 7, (b, 12, n)).float()
    return x


def n_params(m):
    return sum(p.numel() for p in m.parameters())


for variant in ["full", "factorised", "generated"]:
    m = STAEformerVar(variant=variant, **ARGS)
    if variant == "generated":
        m.set_descriptors(torch.rand(170, 48))
    y = m(fake_batch(4, 170))
    y.mean().backward()
    print(f"{variant:11s} params={n_params(m):,} out={tuple(y.shape)}")

# inductive check: train on 136 sensors, then run on all 170
args = dict(ARGS, num_nodes=136)
m = STAEformerVar(variant="generated", **args)
m.set_descriptors(torch.rand(136, 48))
m(fake_batch(4, 136)).mean().backward()
assert m.node_mlp[0].weight.grad is not None
m.eval()
m.set_descriptors(torch.rand(170, 48))
with torch.no_grad():
    y = m(fake_batch(4, 170))
print("inductive forward OK:", tuple(y.shape))
