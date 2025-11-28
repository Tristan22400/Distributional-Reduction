import json
import sys
import os
import argparse

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.replication import plot_prototype_evolution, plot_tradeoff_analysis

def main():
    parser = argparse.ArgumentParser(description="Plot results from a JSON file.")
    parser.add_argument("--file", type=str, default="results.json", help="Path to the results JSON file.")
    parser.add_argument("--output_evolution", type=str, default="multi_dataset_evolution_from_json.png", help="Output filename for evolution plot.")
    parser.add_argument("--output_tradeoff", type=str, default="tradeoff_analysis_from_json.png", help="Output filename for tradeoff plot.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    print(f"Loading results from '{args.file}'...")
    with open(args.file, 'r') as f:
        data = json.load(f)

    # Extract data
    prototype_counts = data.get("prototype_counts")
    methods = data.get("methods")
    metrics = data.get("metrics")
    results = data.get("results")

    if not results:
        print("Error: No results found in the JSON file.")
        return

    print(f"Datasets found: {list(results.keys())}")
    print(f"Methods: {methods}")
    print(f"Metrics: {metrics}")
    print(f"Prototype counts: {prototype_counts}")

    # Plotting
    print(f"\nGenerating evolution plot: {args.output_evolution}...")
    plot_prototype_evolution(
        results,
        prototype_counts,
        methods=methods,
        metrics=metrics,
        filename=args.output_evolution
    )

    print(f"\nGenerating trade-off analysis plot: {args.output_tradeoff}...")
    plot_tradeoff_analysis(
        results,
        methods=methods,
        filename=args.output_tradeoff
    )

    print("\nDone.")

if __name__ == "__main__":
    main()
