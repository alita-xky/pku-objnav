
import os
import csv
import json
import subprocess
from datetime import datetime


SCENES = [
    "FloorPlan201",
    "FloorPlan202",
    "FloorPlan203",
    "FloorPlan204",
    "FloorPlan205",
]

TARGETS = [
    "plant",
    "coffee table",
    "book",
    "laptop",
    "remote control",
]


def safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_")


def run_one(scene, target, output_root, device="cpu", max_goals=50):
    exp_name = f"{scene}_{safe_name(target)}"
    save_dir = os.path.join(output_root, exp_name)
    result_json = os.path.join(save_dir, "result.json")

    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python",
        "-u",
        "run_spub_ovnav.py",
        "--scene",
        scene,
        "--target",
        target,
        "--save_dir",
        save_dir,
        "--result_json",
        result_json,
        "--device",
        device,
        "--max_goals",
        str(max_goals),
        "--quiet",
    ]

    print(f"\n[RUN] scene={scene}, target={target}")
    print(" ".join(cmd))

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=None,
        )

        console_log = completed.stdout

        with open(os.path.join(save_dir, "console.log"), "w") as f:
            f.write(console_log)

        if completed.returncode != 0:
            print("[FAILED PROCESS]")
            print(console_log[-1000:])

            return {
                "scene": scene,
                "target": target,
                "success": False,
                "error": "process_failed",
                "returncode": completed.returncode,
                "save_dir": save_dir,
            }

        if not os.path.exists(result_json):
            print("[FAILED RESULT] result.json not found")

            return {
                "scene": scene,
                "target": target,
                "success": False,
                "error": "missing_result_json",
                "returncode": completed.returncode,
                "save_dir": save_dir,
            }

        with open(result_json, "r") as f:
            result = json.load(f)

        result["error"] = ""
        result["returncode"] = completed.returncode

        print(
            f"[OK] success={result.get('success')}, "
            f"actions={result.get('total_actions')}, "
            f"goals={result.get('visited_belief_goals')}, "
            f"anchors={result.get('anchor_searches')}"
        )

        return result

    except Exception as e:
        print("[EXCEPTION]", repr(e))

        return {
            "scene": scene,
            "target": target,
            "success": False,
            "error": repr(e),
            "returncode": -1,
            "save_dir": save_dir,
        }


def write_summary_csv(results, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fields = [
        "scene",
        "target",
        "resolved_target",
        "success",
        "visited_belief_goals",
        "visited_reachable_points",
        "total_actions",
        "num_context_objects",
        "anchor_searches",
        "searched_anchors",
        "error",
        "returncode",
        "save_dir",
        "log_path",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)


def print_table(results):
    print("\n========== Batch Summary ==========")
    print(
        f"{'Scene':<14} "
        f"{'Target':<16} "
        f"{'Success':<8} "
        f"{'Actions':<8} "
        f"{'Goals':<6} "
        f"{'Anchors':<7} "
        f"{'Error'}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r.get('scene', ''):<14} "
            f"{r.get('target', ''):<16} "
            f"{str(r.get('success', '')):<8} "
            f"{str(r.get('total_actions', '')):<8} "
            f"{str(r.get('visited_belief_goals', '')):<6} "
            f"{str(r.get('anchor_searches', '')):<7} "
            f"{r.get('error', '')}"
        )

    total = len(results)
    success_count = sum(1 for r in results if r.get("success") is True)

    print("-" * 90)
    print(f"Success rate: {success_count}/{total} = {success_count / max(total, 1):.2%}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = f"outputs/batch_spub_{timestamp}"
    os.makedirs(output_root, exist_ok=True)

    results = []

    for scene in SCENES:
        for target in TARGETS:
            result = run_one(
                scene=scene,
                target=target,
                output_root=output_root,
                device="cpu",
                max_goals=50,
            )
            results.append(result)

            output_csv = os.path.join(output_root, "summary.csv")
            write_summary_csv(results, output_csv)
            print_table(results)

    output_csv = os.path.join(output_root, "summary.csv")
    write_summary_csv(results, output_csv)

    print("\nSaved batch results to:")
    print(output_csv)
    print(output_root)


if __name__ == "__main__":
    main()
