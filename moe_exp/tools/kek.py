import pandas as pd

configs = [
    ("r8_b96", "moe_exp/read_plan_cumulative_r8_b96_gap0.csv", 8),
    ("r16_b96", "moe_exp/read_plan_cumulative_r16_b96_gap0.csv", 16),
]

rows = []

for name, path, refresh_every in configs:
    df = pd.read_csv(path)

    out = {
        "config": name,
        "refresh_every": refresh_every,
    }

    for kind in ["initial_reload", "refresh_reload", "cold"]:
        x = df[df.kind == kind]
        g = x.groupby(["run_name", "decode_pos"])

        out[f"{kind}_groups"] = len(g)
        out[f"{kind}_mib_group"] = g["nbytes"].sum().mean() / 1024**2 if len(g) else 0.0
        out[f"{kind}_ranges_group"] = g.size().mean() if len(g) else 0.0

    x = df[df.kind == "refresh_reload"]
    g = x.groupby(["run_name", "decode_pos"])

    if len(g):
        out["refresh_mib_per_decode_token"] = (
            g["nbytes"].sum().mean() / 1024**2 / refresh_every
        )
        out["refresh_ranges_per_decode_token"] = g.size().mean() / refresh_every
    else:
        out["refresh_mib_per_decode_token"] = 0.0
        out["refresh_ranges_per_decode_token"] = 0.0

    # Approx critical/background split.
    out["blocking_cold_mib_group"] = out["cold_mib_group"]
    out["blocking_cold_ranges_group"] = out["cold_ranges_group"]
    out["background_refresh_mib_token"] = out["refresh_mib_per_decode_token"]
    out["background_refresh_ranges_token"] = out["refresh_ranges_per_decode_token"]

    rows.append(out)

summary = pd.DataFrame(rows)

cols = [
    "config",
    "refresh_every",
    "initial_reload_mib_group",
    "initial_reload_ranges_group",
    "cold_groups",
    "cold_mib_group",
    "cold_ranges_group",
    "refresh_reload_groups",
    "refresh_reload_mib_group",
    "refresh_reload_ranges_group",
    "refresh_mib_per_decode_token",
    "refresh_ranges_per_decode_token",
]

print(summary[cols].to_string(index=False))

print("\nDelta r16 - r8:")
a = summary.set_index("config")
delta = a.loc["r16_b96"] - a.loc["r8_b96"]
for k, v in delta.items():
    if k == "refresh_every":
        continue
    print(f"{k:36s} {v:10.3f}")
