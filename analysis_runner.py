#!/usr/bin/env python3
"""
Batch Experiment Analysis Runner with HPC support
-------------------------------------------------
Supports 'usplit' and 'microsplit' folder structures.
Can run locally or submit to HPC via SLURM.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import typer

app = typer.Typer(help="Run batch experiment analysis locally or on HPC")

# -------------------------------------------------------------------
# 🔹 Helpers
# -------------------------------------------------------------------
def find_prediction_pairs(results_base: Path, model_name: str):
    """Find SW–OG prediction pairs for usplit or microsplit structure."""
    pairs = []

    if model_name == "usplit":
        for dataset_dir in results_base.iterdir():
            if not dataset_dir.is_dir(): 
                continue
            for modality_dir in dataset_dir.iterdir():
                if not modality_dir.is_dir():
                    continue
                for lc_dir in modality_dir.iterdir():
                    if not lc_dir.is_dir():
                        continue
                    og_files = list(lc_dir.glob("*_og.pkl*"))
                    sw_files = list(lc_dir.glob("*_swt.pkl*"))
                    if og_files and sw_files:
                        pairs.append({
                            "dataset": dataset_dir.name,
                            "modality": modality_dir.name,
                            "lc": lc_dir.name,
                            "og_path": str(og_files[0]),
                            "sw_path": str(sw_files[0]),
                        })
    elif model_name == "microsplit":
        for dataset_dir in results_base.iterdir():
            if not dataset_dir.is_dir():
                continue
            og_files = list(dataset_dir.glob("*_og.pkl*"))
            sw_files = list(dataset_dir.glob("*_swt.pkl*"))
            if og_files and sw_files:
                pairs.append({
                    "dataset": dataset_dir.name,
                    "og_path": str(og_files[0]),
                    "sw_path": str(sw_files[0]),
                })

    return pairs


def select_pairs_interactively(pairs):
    """Interactive CLI for selecting which SW–OG pairs to analyze."""
    print("\nAvailable SW–OG prediction pairs:")
    for i, p in enumerate(pairs, 1):
        label = f"{p['dataset']}"
        if "modality" in p:
            label += f"/{p['modality']}"
        if "lc" in p:
            label += f"/{p['lc']}"
        print(f"{i}: {label}")
    
    selection = input("\nEnter numbers of the pairs to run (comma-separated) or [A] for all:\n> ").strip()
    if selection.upper() == "A":
        return pairs
    
    indices = [int(s) - 1 for s in selection.split(",") if s.strip().isdigit()]
    selected = [pairs[i] for i in indices if 0 <= i < len(pairs)]
    if not selected:
        print("❌ No valid selections made. Exiting.")
        raise typer.Exit()
    return selected


def build_analysis_cmd(args, pair):
    """Build the Python command list for analysis."""
    if os.getcwd() == str(Path(args["project_dir"])):
        analyze_script = "analysis/analyze_experiment.py"
    else:
        analyze_script = Path(args["project_dir"]) / "analysis" / "analyze_experiment.py"
    save_dir = Path(args["save_base"]) / pair["dataset"]
    if "modality" in pair: save_dir /= pair["modality"]
    if "lc" in pair: save_dir /= pair["lc"]

    cmd = [
        str(args["python_bin"]), str(analyze_script),
        "--model_name", args["model_name"],
        "--dataset", pair["dataset"],
        "--pred_sw", pair["sw_path"],
        "--pred_og", pair["og_path"],
        "--save_dir", str(save_dir),
        "--bins", args.get("bins", "50"),
        "--channel", "all",
        "--kl_start", "29",
        "--kl_end", "33",
    ]
    if pair["dataset"] == "HT_H24":
        cmd.append(f"--inner_tile_size")
        cmd.append(args.get('inner_tile_size', "9,32,32"))
    else :
        cmd.append(f"--inner_tile_size")
        cmd.append(args.get('inner_tile_size', "32,32"))

    if args["gradient_based_analysis"]:
        cmd.append("--gradient_based_analysis")
        cmd.append("True")
        
    if args["qualitative_analysis"]:
        cmd.append("--qualitative_analysis")
        cmd.append("True")
    # if args["all"]:
    #     cmd.append("--all")

    return cmd


def run_local(pair, args):
    """Run analysis locally."""
    save_dir = Path(args["save_base"]) / pair["dataset"]
    if "modality" in pair: save_dir /= pair["modality"]
    if "lc" in pair: save_dir /= pair["lc"]
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_analysis_cmd(args, pair)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = save_dir / f"analysis_log_{timestamp}.log"

    if args["dry_run"]:
        print("[DRY RUN]", " ".join(cmd))
        return f"[SKIPPED] {pair}"

    print(f"▶ Running local analysis for {pair['dataset']}")
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"✅ Completed {pair['dataset']}")
    return f"[DONE] {pair['dataset']}"


def run_hpc(pair, args):
    """Submit job to HPC via SLURM."""
    save_dir = Path(args["save_base"]) / pair["dataset"]
    if "modality" in pair: save_dir /= pair["modality"]
    if "lc" in pair: save_dir /= pair["lc"]
    
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd_str = " ".join(build_analysis_cmd(args, pair))
    jobname = f"grad_{pair['dataset']}"
    sbatch_file = save_dir / f"sbatch_{jobname}.sh"

    sbatch_contents = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={save_dir}/hpc_{jobname}.log
#SBATCH --error={save_dir}/hpc_{jobname}_err.log
#SBATCH --partition={args['partition']}
#SBATCH --gres=gpu:{args['gpus']}
#SBATCH --mem={args['mem']}
#SBATCH --cpus-per-task={args['cpus']}
#SBATCH --time={args['time']}
source ~/.bashrc
conda activate msr
cd {args['project_dir']}
{cmd_str}
"""
    sbatch_file.write_text(sbatch_contents)

    if not args["dry_run"]:
        subprocess.run(f"ssh hpc 'bash -l -c \"sbatch {sbatch_file}\"'", shell=True, check=True)

    return f"[SUBMITTED] {pair['dataset']} ({sbatch_file})"


# -------------------------------------------------------------------
# 🔹 Main entry point
# -------------------------------------------------------------------
@app.command()
def main(
    model_name: str = typer.Option("microsplit", help="Model name: 'usplit' or 'microsplit'"),
    results_base: Path = typer.Option(..., help="Base folder with results to analyze"),
    project_dir: Path = typer.Option('/home/aman.kukde/sliding_windowed_tiling/', help="Path to main project dir"),
    save_base: Path = typer.Option(..., help="Where to save analysis outputs"),
    python_bin: Path = typer.Option("python3", help="Python executable path"),
    max_workers: int = typer.Option(4, help="Number of parallel local jobs"),
    dry_run: bool = typer.Option(False, help="Print commands without running"),
    gradient_based_analysis: bool = typer.Option(True),
    qualitative_analysis: bool = typer.Option(True),
    all: bool = typer.Option(True, help="Run all analysis steps"),
    hpc: bool = typer.Option(False, help="Run via HPC instead of locally"),
    partition: str = typer.Option("gpuq"),
    gpus: int = typer.Option(1),
    mem: str = typer.Option("64GB"),
    cpus: int = typer.Option(4),
    time: str = typer.Option("12:00:00"),
    interactive: bool = typer.Option(True, help="Prompt user to select which pairs to run"),
):
    args = locals()  # Pass around as dict

    print(f"🔍 Scanning for prediction pairs in {results_base} ...")
    pairs = find_prediction_pairs(results_base, model_name)
    print(f"Found {len(pairs)} valid SW–OG pairs.")
    if not pairs:
        typer.echo("No valid pairs found. Exiting.")
        raise typer.Exit()

    if interactive:
        selected_pairs = select_pairs_interactively(pairs)
    else:
        selected_pairs = pairs

    print(f"Selected {len(selected_pairs)} pairs for analysis.\n")
    results = []

    if hpc:
        for p in selected_pairs:
            results.append(run_hpc(p, args))
    
    else:
        # Single-threaded version (no multiprocessing)
        for p in selected_pairs:
            result = run_local(p, args)
            results.append(result)
    # else:
    #     with ProcessPoolExecutor(max_workers=max_workers) as ex:
    #         futures = [ex.submit(run_local, p, args) for p in selected_pairs]
    #         for f in as_completed(futures):
    #             results.append(f.result())

    log_dir = save_base / "analysis_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"run_log_{timestamp}.log"
    log_file.write_text("\n".join(results))
    typer.echo(f"\n📝 Log saved to: {log_file}\n✅ Done.")

if __name__ == "__main__":
    app()
