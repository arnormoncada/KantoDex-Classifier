import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

sns.set_theme(style="whitegrid", font_scale=1.1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard data into professional multi-subplot and class-accuracy plots.",
    )
    parser.add_argument(
        "--events_file",
        type=str,
        required=True,
        help="Path to the TensorBoard events.out.tfevents.* file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tb_plots",
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="",
        help="Optional prefix for plot titles.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    events_path = args.events_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading events file: {events_path}")
    ea = event_accumulator.EventAccumulator(
        events_path,
        size_guidance={
            event_accumulator.SCALARS: 0,  # load all scalar data
        },
    )
    ea.Reload()

    all_scalar_tags = ea.Tags().get("scalars", [])
    print(f"Found scalar tags:\n{all_scalar_tags}")

    # ----------------------
    # 1) Multi-subplot figure for key metrics
    # ----------------------
    plot_config = [
        {
            "title": "Loss",
            "train_tag": "Epoch Training Loss",
            "val_tag": "Epoch Validation Loss",
            "ylabel": "Loss",
        },
        {
            "title": "Accuracy",
            "train_tag": "Epoch Training Accuracy",
            "val_tag": "Epoch Validation Accuracy",
            "ylabel": "Accuracy (%)",
        },
    ]

    # We'll gather data for these tags
    panel_data = {}  # { 'tag': (steps[], values[]) }

    # Helper function
    def get_scalar_data(tag: str):
        """Return (steps, values) for a given scalar tag. If not found, returns (None, None)."""
        if tag in all_scalar_tags:
            scalar_events = ea.Scalars(tag)
            steps = [s.step for s in scalar_events]
            vals = [s.value for s in scalar_events]
            return steps, vals
        return None, None

    # Collect data
    for pcfg in plot_config:
        tr_tag = pcfg["train_tag"]
        val_tag = pcfg["val_tag"]
        tr_steps, tr_vals = get_scalar_data(tr_tag)
        if val_tag:
            val_steps, val_vals = get_scalar_data(val_tag)
        else:
            val_steps, val_vals = None, None

        panel_data[pcfg["title"]] = {
            "train": (tr_steps, tr_vals),
            "val": (val_steps, val_vals),
            "ylabel": pcfg["ylabel"],
        }

    # Set up Seaborn style
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Create subplots
    _, axes = plt.subplots(
        nrows=len(plot_config),
        ncols=1,
        figsize=(10, len(plot_config) * 4),
        sharex=False,
    )
    if len(plot_config) == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot

    for ax, cfg in zip(axes, plot_config, strict=False):
        metric_title = cfg["title"]
        train_data = panel_data[metric_title]["train"]
        val_data = panel_data[metric_title]["val"]
        ylabel = panel_data[metric_title]["ylabel"]

        # Plot training data
        if train_data[0] is not None:
            sns.lineplot(
                x=train_data[0],
                y=train_data[1],
                ax=ax,
                label="Training",
                marker="o",
                markersize=6,
                linewidth=2,
            )

        # Plot validation data
        if val_data[0] is not None:
            sns.lineplot(
                x=val_data[0],
                y=val_data[1],
                ax=ax,
                label="Validation",
                marker="s",
                markersize=6,
                linewidth=2,
            )

        ax.set_title(f"{args.title_prefix}{metric_title}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    # Set common X-label
    axes[-1].set_xlabel("Epoch")

    plt.tight_layout()
    multi_fig_path = os.path.join(output_dir, "combined_metrics.png")
    plt.savefig(multi_fig_path, dpi=300)
    plt.close()
    print(f"Saved combined training/validation metrics plot: {multi_fig_path}")

    # ----------------------
    # 2) Per-class accuracy histogram
    # ----------------------

    # Tags for class accuracy usually look like 'Class Accuracy/<pokemon_name>'
    # Let's gather them and store them in a dict { pokemon_name: final_accuracy }
    class_acc_tags = [t for t in all_scalar_tags if t.startswith("Class Accuracy/")]

    class_acc_map = {}  # { 'Bulbasaur': accuracy_value, ... }

    for catag in class_acc_tags:
        # We assume the last logged value is the final accuracy
        # Retrieve entire time-series
        scalar_events = ea.Scalars(catag)
        final_value = scalar_events[-1].value  # last entry
        # Extract class name from tag. e.g. "Class Accuracy/Bulbasaur" -> "Bulbasaur"
        class_name = catag.split("Class Accuracy/")[-1]
        class_acc_map[class_name] = final_value

    if len(class_acc_map) == 0:
        print("No per-class accuracy tags found. Skipping class accuracy plot.")
        return

    # Extract accuracies
    accuracies = list(class_acc_map.values())

    # Use seaborn to create a better-looking histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(accuracies, bins=20, kde=True, color="skyblue", edgecolor="black")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of Classes")
    plt.title(f"{args.title_prefix}Histogram of Per-Class Accuracies")
    plt.tight_layout()

    hist_fig_path = os.path.join(output_dir, "per_class_accuracy_histogram.png")
    plt.savefig(hist_fig_path, dpi=150)
    plt.close()
    print(f"Saved histogram of per-class accuracies: {hist_fig_path}")

    # Get best overall accuracy and corresponding loss
    if "Epoch Validation Accuracy" in all_scalar_tags:
        tag = "Epoch Validation Accuracy"
        scalar_events = ea.Scalars(tag)
        best_acc = max([s.value for s in scalar_events])
        best_acc_epoch = next(s.step for s in scalar_events if s.value == best_acc)
        best_acc_loss = next(
            s.value for s in ea.Scalars("Epoch Validation Loss") if s.step == best_acc_epoch
        )
        print(
            f"Best Validation Accuracy: {best_acc:.2f}% at Epoch {best_acc_epoch}, Loss: {best_acc_loss:.4f}",
        )

    print("All done.")


if __name__ == "__main__":
    main()
