import os
import json
import pandas as pd
import numpy as np

def compile_data():
    print("Compiling dashboard datasets...")
    
    # 1. Load robustness metrics
    rob_path = "results/metrics/robustness_sweep_results.csv"
    robustness_data = []
    if os.path.exists(rob_path):
        df = pd.read_csv(rob_path)
        robustness_data = df.to_dict(orient="records")
        
    # 2. Load cross-dataset metrics
    cross_path = "results/metrics/cross_dataset_results.csv"
    cross_data = []
    if os.path.exists(cross_path):
        df = pd.read_csv(cross_path)
        cross_data = df.to_dict(orient="records")
        
    # 3. Load sensitivity metrics
    sens_path = "results/metrics/sensitivity_results.csv"
    sensitivity_data = []
    if os.path.exists(sens_path):
        df = pd.read_csv(sens_path)
        sensitivity_data = df.to_dict(orient="records")
        
    # 4. Load statistical metrics
    stats_path = "results/metrics/statistical_results.json"
    statistical_data = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            statistical_data = json.load(f)

    # 4.5 Load baseline benchmarks
    baseline_path = "results/metrics/baseline_results.json"
    baseline_data = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)

    # 5. Load dynamic interactive graph data (SKAB_SAMPLE, BATADAL_SAMPLE, EXPLAIN_DATA)
    interactive_path = "results/metrics/dashboard_interactive_data.json"
    interactive_data = {}
    if os.path.exists(interactive_path):
        with open(interactive_path, "r") as f:
            interactive_data = json.load(f)
            
    skab_sample = interactive_data.get("SKAB_SAMPLE", [])
    batadal_sample = interactive_data.get("BATADAL_SAMPLE", [])
    explain_data = interactive_data.get("EXPLAIN_DATA", {})

    # Write output JavaScript file
    os.makedirs("dashboard", exist_ok=True)
    js_content = f"""// Auto-generated data store for Explainable Time Series Automata Dashboard
const ROBUSTNESS_DATA = {json.dumps(robustness_data, indent=2)};
const CROSS_DATASET_DATA = {json.dumps(cross_data, indent=2)};
const SENSITIVITY_DATA = {json.dumps(sensitivity_data, indent=2)};
const STATISTICAL_DATA = {json.dumps(statistical_data, indent=2)};
const BASELINE_DATA = {json.dumps(baseline_data, indent=2)};
const SKAB_SAMPLE = {json.dumps(skab_sample, indent=2)};
const BATADAL_SAMPLE = {json.dumps(batadal_sample, indent=2)};
const EXPLAIN_DATA = {json.dumps(explain_data, indent=2)};
"""

    with open("dashboard/data_store.js", "w") as f:
        f.write(js_content)
    print("[SUCCESS] Successfully generated dashboard/data_store.js!")

if __name__ == "__main__":
    compile_data()
