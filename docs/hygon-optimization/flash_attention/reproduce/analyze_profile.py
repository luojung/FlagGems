import argparse
import collections
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("profile", type=Path)
args = parser.parse_args()
root = Path.cwd()
ev = json.loads(args.profile.read_text())["traceEvents"]
rois = [
    e
    for e in ev
    if e.get("cat") == "HIPTX"
    and e.get("name", "").startswith("flash:")
    and ":roi=" in e["name"]
]
dev = [e for e in ev if e.get("cat") == "HIPOPS" and e.get("ph") == "X"]
arms = {}
for r in rois:
    name = r["name"].split(":")[1]
    k = [
        e
        for e in dev
        if r["ts"] <= e["ts"] and e["ts"] + e["dur"] <= r["ts"] + r["dur"]
    ]
    assert len(k) == (30 if name.startswith(("repo", "hip")) else 10), (name, len(k))
    arms.setdefault(name, []).append(
        {
            "roi": r["name"],
            "calls": 10,
            "kernels": dict(collections.Counter(e["name"] for e in k)),
            "us": sum(e["dur"] for e in k) / 10,
        }
    )
assert len(arms) == 6, len(arms)
timing = {
    key: statistics.median(
        statistics.median(x["us"] for x in arms[name])
        for name in [key, key + "_repeat"]
    )
    for key in ["wheel", "aten", "repo"]
}
v = json.loads((root / "repo_final_profile_validation.json").read_text())
result = {
    "method": "hipprof HIPOPS duration; all helper kernels included; 5 ROIs x 10 calls x 2 repeats",
    "timing_us": timing,
    "wheel_over_repo": timing["wheel"] / timing["repo"],
    "meets_90": timing["wheel"] / timing["repo"] >= 0.9,
    "source_sha256": v["source_sha256"],
    "arms": arms,
}
(root / "repo_final_kernel_summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "arms"}, indent=2))
