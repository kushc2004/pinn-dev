"""Score the candidate physics laws alone against the held-out envelope."""
import json

import numpy as np
import torch

from src import data, physics


def r2(t, p):
    return 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)


split = data.make_split("envelope")
x = torch.from_numpy(split.x_train)
xt = torch.from_numpy(split.x_test)

eq = json.load(open("results/equation.json"))
laws = {"sr_discovered": physics.compile_tree(eq["tree"]), "legacy": physics.legacy_physics_fn()}
print(f"Held-out envelope test set (n={len(xt)}):")
for name, fn in laws.items():
    with torch.no_grad():
        pred_tr = fn(x).numpy()
        pred_te = fn(xt).numpy()
    mu, sd = pred_tr.mean(), pred_tr.std()
    z = (pred_te - mu) / max(sd, 1e-8)
    print(f"  {name}: standardized-law R2 on envelope = {r2(split.y_test, z):.3f}")
