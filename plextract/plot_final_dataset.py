import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_final_dataset(final_csv_path: str, save_path: str = None):
    """
    Plot all series from the final_dataset.csv file and save to PNG.

    Automatically uses canonicalized values (x_value, y_value) and
    labeled axes with units when available.

    Parameters
    ----------
    final_csv_path : str
        Path to the final_dataset.csv file.
    save_path : str | None
        Optional output image path. If None, saved as 'final_plot.png'
        in the same directory as the CSV.
    """
    if not os.path.exists(final_csv_path):
        raise FileNotFoundError(f"Could not find {final_csv_path}")

    df = pd.read_csv(final_csv_path)

    # ---- Determine which columns to use ----
    use_converted = {"x_value", "y_value"}.issubset(df.columns)
    x_col = "x_value" if use_converted else "x"
    y_col = "y_value" if use_converted else "y"

    # ---- Determine axis labels ----
    if {"x_name", "x_unit"}.issubset(df.columns):
        x_name = df["x_name"].dropna().unique()
        x_label = x_name[0] if len(x_name) == 1 else "X"
        x_unit = df["x_unit"].dropna().unique()
        x_label = f"{x_label} [{x_unit[0]}]" if len(x_unit) == 1 else x_label
    else:
        x_label = "X value"

    if {"y_name", "y_unit"}.issubset(df.columns):
        y_name = df["y_name"].dropna().unique()
        y_label = y_name[0] if len(y_name) == 1 else "Y"
        y_unit = df["y_unit"].dropna().unique()
        y_label = f"{y_label} [{y_unit[0]}]" if len(y_unit) == 1 else y_label
    else:
        y_label = "Y value"

    # ---- Validate columns ----
    if not {"series_label", x_col, y_col}.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: series_label, {x_col}, {y_col}"
        )

    # ---- Determine save path ----
    if save_path is None:
        save_path = os.path.join(os.path.dirname(final_csv_path), "final_plot.png")

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    for label, group in df.groupby("series_label"):
        group = group.sort_values(x_col)
        plt.plot(group[x_col], group[y_col], label=label)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title("Extracted and Labeled Series", fontsize=14, weight="bold")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[final plot] Saved: {save_path}")