import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def plot_cnodes_evolution(file_name, window=None, only_clients=True,
                          output_svg=True, show=False):
    # Load goodness_0.txt
    df = pd.read_csv(file_name, sep="\t")

    # Optionally filter only client worlds
    if only_clients:
        df = df[df["World"].astype(str).str.startswith("client_")]

    # Sort by iteration
    df = df.sort_values("Iteration").reset_index(drop=True)

    # Ensure numeric types
    df["C-nodes"] = pd.to_numeric(df["C-nodes"], errors="coerce").fillna(0)
    df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce")

    # Define smoothing window if not provided
    if window is None:
        # Roughly 1/100 of total length, minimum 50
        window = max(50, len(df) // 100)

    df["Cnodes_rolling"] = (
        df["C-nodes"].rolling(window=window, min_periods=1).mean()
    )

    # Convert to numpy arrays (avoids pandas/matplotlib indexing issues)
    iters = df["Iteration"].to_numpy()
    cnodes = df["C-nodes"].to_numpy()
    cnodes_roll = df["Cnodes_rolling"].to_numpy()

    # Create figure (bigger for paper screenshots)
    plt.figure(figsize=(12, 7))
    plt.title("Evolution of the number of C-nodes (bartender)", fontsize=20)
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Number of C-nodes", fontsize=16)

    # Raw line (very transparent)
    plt.plot(
        iters,
        cnodes,
        color="gray",
        alpha=0.2,
        linewidth=1.5,
        label="C-nodes (raw)",
    )

    # Moving average
    plt.plot(
        iters,
        cnodes_roll,
        color="C0",
        linewidth=3,
        label=f"Moving average (window = {window})",
    )

    plt.grid(linewidth=0.6, alpha=0.5)
    plt.legend(loc="best", fontsize=14)

    # Larger tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    base_file_name, _ = os.path.splitext(file_name)
    ext = "svg" if output_svg else "png"
    out_name = f"{base_file_name}_cnodes_evolution.{ext}"

    if os.path.exists(out_name):
        print(f"File already exists: {out_name}")
    else:
        plt.savefig(out_name, dpi=300)  # high DPI for publication
        print(f"Saved: {out_name}")

    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evolution of the number of C-nodes from goodness_0.txt"
    )
    parser.add_argument(
        "-f", "--file", required=True,
        help="goodness_0.txt file from the bartender experiment",
    )
    parser.add_argument(
        "-w", "--window", type=int, default=None,
        help="Window size for the moving average (default: ~1/100 of the log length)",
    )
    parser.add_argument(
        "--all_worlds", action="store_true",
        help="Include all worlds (by default only client_*)",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="Save output as PNG instead of SVG",
    )
    parser.add_argument(
        "-s", "--show", action="store_true",
        help="Show the figure after generating it",
    )

    args = parser.parse_args()

    plot_cnodes_evolution(
        file_name=args.file,
        window=args.window,
        only_clients=not args.all_worlds,
        output_svg=not args.png,
        show=args.show,
    )


if __name__ == "__main__":
    main()
