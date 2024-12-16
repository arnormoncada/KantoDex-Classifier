import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

sns.set_theme(style="whitegrid", font_scale=1.2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export specific scalar metrics from a TensorBoard events file into combined plots.",
    )
    parser.add_argument(
        "--events_file",
        type=str,
        required=True,
        help="Path to the TensorBoard events.out.tfevents file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tb_plots",
        help="Directory to save the combined plots (default: tb_plots/).",
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

    # Initialize the EventAccumulator to load scalar data.
    print(f"Loading events file: {events_path}")
    ea = event_accumulator.EventAccumulator(
        events_path,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        },
    )
    ea.Reload()

    # Identify which tags are present:
    all_scalar_tags = ea.Tags().get("scalars", [])
    print(f"Found scalar tags: {all_scalar_tags}")

    # Cherry-pick tags of interest (adjust these names to match your logs)
    desired_loss_tags = ["Epoch Training Loss", "Epoch Validation Loss"]
    desired_accuracy_tags = ["Epoch Training Accuracy", "Epoch Validation Accuracy"]

    # Filter out only those tags that actually exist in the event file
    loss_tags = [tag for tag in desired_loss_tags if tag in all_scalar_tags]
    acc_tags = [tag for tag in desired_accuracy_tags if tag in all_scalar_tags]

    # Container to store data: { "tag": (steps, values) }
    loss_data = {}
    acc_data = {}

    # Extract data for each tag
    for tag in loss_tags:
        scalar_events = ea.Scalars(tag)
        steps = [s.step for s in scalar_events]
        values = [s.value for s in scalar_events]
        loss_data[tag] = (steps, values)

    for tag in acc_tags:
        scalar_events = ea.Scalars(tag)
        steps = [s.step for s in scalar_events]
        values = [s.value for s in scalar_events]
        acc_data[tag] = (steps, values)

    # --- Plot 1: Combined Loss ---
    if loss_data:
        plt.figure(figsize=(8, 5))
        for tag, (steps, values) in loss_data.items():
            plt.plot(steps, values, marker="o", markersize=3, linewidth=2, label=tag)
        plt.title(f"{args.title_prefix}Loss Curves")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        loss_out_path = os.path.join(output_dir, "combined_loss.png")
        plt.savefig(loss_out_path, dpi=150)
        plt.close()
        print(f"Saved combined loss plot: {loss_out_path}")
    else:
        print("No loss tags found in events file for combined plot.")

    # --- Plot 2: Combined Accuracy ---
    if acc_data:
        plt.figure(figsize=(8, 5))
        for tag, (steps, values) in acc_data.items():
            plt.plot(steps, values, marker="o", markersize=3, linewidth=2, label=tag)
        plt.title(f"{args.title_prefix}Accuracy Curves")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        acc_out_path = os.path.join(output_dir, "combined_accuracy.png")
        plt.savefig(acc_out_path, dpi=150)
        plt.close()
        print(f"Saved combined accuracy plot: {acc_out_path}")
    else:
        print("No accuracy tags found in events file for combined plot.")

    print("All done.")


if __name__ == "__main__":
    main()
