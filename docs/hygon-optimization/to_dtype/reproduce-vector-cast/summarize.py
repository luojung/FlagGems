import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = []
for suite in ("model", "scale"):
    results = json.loads((root / f"{suite}-sqlite-summary.json").read_text())["results"]
    cases = {}
    for measurement in results:
        cases.setdefault(measurement["case"], {})[measurement["provider"]] = measurement
    for row, providers in sorted(cases.items()):
        das = providers.pop("das")
        baseline = providers.pop("new_public")
        assert len(providers) <= 1
        fg = next(iter(providers.values()), baseline)
        ratios = []
        if not fg["samples"][0]["noop"]:
            for group in range(3):
                if suite == "model":
                    ratios.append(
                        das["aa_medians_us"][group] / fg["aa_medians_us"][group]
                    )
                else:

                    def durations(result):
                        return {
                            r["roi"]: r["device_us"]
                            for r in result["samples"]
                            if r["aa"] == group
                        }

                    a, b = durations(das), durations(fg)
                    assert a.keys() == b.keys() and len(a) == 5
                    ratios.append(statistics.mean(a[k] / b[k] for k in a))
        minimum = min(ratios, default=None)
        if not fg["correctness"]:
            status = "FG_FAIL"
        elif not ratios:
            status = "NO_DEVICE_CAST"
        elif not das["correctness"]:
            status = "DAS_INCORRECT"
        else:
            status = "PASS" if minimum >= 0.90 else "BELOW_90"
        output.append(
            dict(
                suite=suite,
                excel_row=row,
                provider=fg["provider"],
                fg_correct=fg["correctness"],
                das_correct=das["correctness"],
                fg_us=fg["device_us"],
                baseline_fg_us=baseline["device_us"],
                das_us=das["device_us"],
                **{f"group_{i + 1}_pct": ratio * 100 for i, ratio in enumerate(ratios)},
                minimum_pct=None if minimum is None else minimum * 100,
                status=status,
            )
        )

fields = (
    "suite",
    "excel_row",
    "provider",
    "fg_correct",
    "das_correct",
    "fg_us",
    "baseline_fg_us",
    "das_us",
    "group_1_pct",
    "group_2_pct",
    "group_3_pct",
    "minimum_pct",
    "status",
)
with (root / "results.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output)

for status in sorted({row["status"] for row in output}):
    print(status, [row["excel_row"] for row in output if row["status"] == status])
