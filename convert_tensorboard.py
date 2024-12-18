import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

sns.set_theme(style="whitegrid", font_scale=1.1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export and compare TensorBoard data from multiple runs."
    )
    parser.add_argument(
        "--events_files",
        type=str,
        required=True,
        help="Comma-separated paths to the TensorBoard events files, e.g. file1,file2",
    )
    parser.add_argument(
        "--names",
        type=str,
        required=True,
        help="Comma-separated names corresponding to each events file, e.g. run1,run2",
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


def load_event_file(events_path):
    """Load a single TensorBoard events file and return an EventAccumulator."""
    ea = event_accumulator.EventAccumulator(
        events_path,
        size_guidance={event_accumulator.SCALARS: 0},  # load all scalar data
    )
    ea.Reload()
    return ea


def get_scalar_data(ea, tag: str):
    """Return (steps, values) for a given scalar tag. If not found, returns (None, None)."""
    all_scalar_tags = ea.Tags().get("scalars", [])
    if tag in all_scalar_tags:
        scalar_events = ea.Scalars(tag)
        steps = [s.step for s in scalar_events]
        vals = [s.value for s in scalar_events]
        return steps, vals
    return None, None


def main():
    args = parse_args()
    events_files = [f.strip() for f in args.events_files.split(",")]
    run_names = [n.strip() for n in args.names.split(",")]

    if len(events_files) != len(run_names):
        raise ValueError("Number of events_files must match number of names.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load all runs
    run_data = {}  # {run_name: {'ea': event_accumulator, 'tags': [...]} }
    for run_name, ef in zip(run_names, events_files):
        print(f"Loading events file for run '{run_name}': {ef}")
        ea = load_event_file(ef)
        run_data[run_name] = {
            "ea": ea,
            "scalars": ea.Tags().get("scalars", []),
        }

    # Configuration for metrics to plot
    # We'll plot these metrics in subplots, and each subplot will have lines from all runs.
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

    # Create multi-subplot figure for key metrics
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(
        nrows=len(plot_config),
        ncols=1,
        figsize=(10, len(plot_config) * 4),
        sharex=False,
    )
    if len(plot_config) == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot

    # For each subplot (metric), plot all runs
    for ax, cfg in zip(axes, plot_config, strict=False):
        metric_title = cfg["title"]
        ylabel = cfg["ylabel"]
        tr_tag = cfg["train_tag"]
        val_tag = cfg["val_tag"]

        for run_name in run_names:
            ea = run_data[run_name]["ea"]
            # Training data
            tr_steps, tr_vals = get_scalar_data(ea, tr_tag)
            # Validation data
            val_steps, val_vals = get_scalar_data(ea, val_tag)

            # Plot training data line
            if tr_steps is not None:
                sns.lineplot(
                    x=tr_steps,
                    y=tr_vals,
                    ax=ax,
                    label=f"{run_name} Train",
                    marker="o",
                    markersize=6,
                    linewidth=2,
                )

            # Plot validation data line
            if val_steps is not None:
                sns.lineplot(
                    x=val_steps,
                    y=val_vals,
                    ax=ax,
                    label=f"{run_name} Val",
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
    multi_fig_path = os.path.join(output_dir, "combined_metrics_comparison.png")
    plt.savefig(multi_fig_path, dpi=300)
    plt.close()
    print(f"Saved combined training/validation metrics comparison plot: {multi_fig_path}")

    # ----------------------
    # Per-class accuracy histograms comparison
    # ----------------------
    # For each run, gather class accuracies. We'll plot them all on one histogram.
    # This requires that "Class Accuracy/<class_name>" tags are present.
    # We'll create a list of accuracy arrays, one per run, and plot them overlapping.
    run_class_accuracies = {}
    all_class_accuracies = []
    for run_name in run_names:
        ea = run_data[run_name]["ea"]
        all_scalar_tags = run_data[run_name]["scalars"]
        class_acc_tags = [t for t in all_scalar_tags if t.startswith("Class Accuracy/")]
        class_acc_map = {}
        for catag in class_acc_tags:
            scalar_events = ea.Scalars(catag)
            final_value = scalar_events[-1].value  # last entry
            class_name = catag.split("Class Accuracy/")[-1]
            class_acc_map[class_name] = final_value
        run_class_accuracies[run_name] = class_acc_map
        if len(class_acc_map) > 0:
            all_class_accuracies.append(list(class_acc_map.values()))

    # If no per-class data available for any run, skip plotting
    if all_class_accuracies:
        # Flatten and determine common bins or just let seaborn handle it
        # We'll plot them in a single figure. Each run's accuracies will be plotted as a separate distribution.
        plt.figure(figsize=(10, 6))
        for run_name in run_names:
            if len(run_class_accuracies[run_name]) > 0:
                accuracies = list(run_class_accuracies[run_name].values())
                sns.histplot(
                    accuracies,
                    bins=20,
                    kde=True,
                    label=run_name,
                    edgecolor="black",
                    alpha=0.3,
                )

        plt.xlabel("Accuracy (%)")
        plt.ylabel("Number of Classes")
        plt.title(f"{args.title_prefix}Histogram of Per-Class Accuracies (All Runs)")
        plt.legend(loc="best")
        plt.tight_layout()

        hist_fig_path = os.path.join(output_dir, "per_class_accuracy_histogram_all_runs.png")
        plt.savefig(hist_fig_path, dpi=150)
        plt.close()
        print(f"Saved histogram of per-class accuracies for all runs: {hist_fig_path}")
    else:
        print("No per-class accuracy tags found for any run. Skipping class accuracy histogram.")

    # Print best validation accuracy info per run
    for run_name in run_names:
        ea = run_data[run_name]["ea"]
        tags = run_data[run_name]["scalars"]
        if "Epoch Validation Accuracy" in tags and "Epoch Validation Loss" in tags:
            val_acc_events = ea.Scalars("Epoch Validation Accuracy")
            best_acc = max([s.value for s in val_acc_events])
            best_acc_epoch = next(s.step for s in val_acc_events if s.value == best_acc)
            val_loss_events = ea.Scalars("Epoch Validation Loss")
            best_acc_loss = next(s.value for s in val_loss_events if s.step == best_acc_epoch)
            print(
                f"Run '{run_name}': Best Validation Accuracy: {best_acc:.2f}% at Epoch {best_acc_epoch}, Loss: {best_acc_loss:.4f}"
            )

    # Print 10 best and worst class accuracies per run
    for run_name in run_names:
        class_acc_map = run_class_accuracies[run_name]
        if len(class_acc_map) == 0:
            continue
        sorted_accs = sorted(class_acc_map.items(), key=lambda x: x[1])
        print(f"\nRun '{run_name}':")
        print(f"Best class accuracies:")
        for i, (cls, acc) in enumerate(sorted_accs[-10:][::-1], 1):
            print(f"{i}. {cls}: {acc:.2f}%")
        print(f"Worst class accuracies:")
        for i, (cls, acc) in enumerate(sorted_accs[:10], 1):
            print(f"{i}. {cls}: {acc:.2f}%")

    print("All done.")


if __name__ == "__main__":
    main()
