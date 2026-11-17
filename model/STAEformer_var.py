import torch
import torch.nn as nn

from STAEformer import STAEformer


class STAEformerVar(STAEformer):
    """STAEformer with a replaceable adaptive embedding.

    variant = "full"       : original free tensor (T x N x d_a), identical to STAEformer
    variant = "factorised" : [u_tau || v_i], free time table + free sensor table
    variant = "generated"  : [u_tau || g(z_i)], sensor part from a descriptor z_i
    """

    def __init__(self, num_nodes, variant="full", time_dim=16,
                 desc_dim=48, hidden=128, **kwargs):
        super().__init__(num_nodes, **kwargs)
        assert variant in ("full", "factorised", "generated")
        self.variant = variant
        if variant == "full":
            return

        d_a = self.adaptive_embedding_dim
        d_n = d_a - time_dim
        assert d_n > 0, "time_dim must be smaller than adaptive_embedding_dim"

        # remove the original free T x N x d_a tensor
        del self.adaptive_embedding

        self.time_table = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(self.in_steps, time_dim))
        )
        if variant == "factorised":
            self.node_table = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(num_nodes, d_n))
            )
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(desc_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d_n),
            )
            self.register_buffer(
                "desc", torch.zeros(num_nodes, desc_dim), persistent=False
            )

    def set_descriptors(self, z):
        """z: (n_nodes, desc_dim) descriptors of the sensors in the current data."""
        self.desc = z.to(self.time_table.device)

    def _build_adaptive(self, n_nodes):
        T = self.in_steps
        u = self.time_table.unsqueeze(1).expand(T, n_nodes, -1)
        if self.variant == "factorised":
            if n_nodes != self.node_table.shape[0]:
                raise ValueError(
                    f"factorised model has {self.node_table.shape[0]} sensor "
                    f"vectors but got {n_nodes} sensors"
                )
            v = self.node_table
        else:
            if self.desc.shape[0] != n_nodes:
                raise ValueError(
                    f"call set_descriptors() with {n_nodes} rows first "
                    f"(have {self.desc.shape[0]})"
                )
            v = self.node_mlp(self.desc)
        v = v.unsqueeze(0).expand(T, -1, -1)
        return torch.cat([u, v], dim=-1)

    def forward(self, x):
        n = x.shape[2]
        # the output projection reshapes with self.num_nodes, so keep it in sync
        # with the sensor count of the current batch (needed for inductive tests)
        self.num_nodes = n
        if self.variant != "full":
            self.adaptive_embedding = self._build_adaptive(n)
        return super().forward(x)
