from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    TensorBoard Logger for PyTorch Models.

    Log various types of data to TensorBoard, including scalars, images, histograms, model graphs,
    and more. It provides methods to add different kind of data.

    Attributes:
        enabled (bool): Flag indicating whether TensorBoard logging is enabled.
        writer (Optional[SummaryWriter]): TensorBoard SummaryWriter instance if enabled, else `None`.

    """

    def __init__(  # noqa: PLR0913
        self,
        log_dir: str | None = None,
        comment: str = "",
        purge_step: int | None = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the TensorBoard Logger.

        Args:
            log_dir (str, optional): Directory to save TensorBoard logs. If `None`, defaults to
                `runs/` with a timestamp.
            comment (str, optional): Comment to append to the log directory name for distinguishing
                 runs.
            purge_step (int, optional): Step from which to purge events in TensorBoard. Useful when
                resuming training.
            max_queue (int, optional): Maximum size of the queue for pending events before forcing
                a flush to disk. Default is 10.
            flush_secs (int, optional): How often (in seconds) to flush pending events to disk.
                Default is 120 seconds.
            filename_suffix (str, optional): Suffix for the TensorBoard event filenames.
            enabled (bool, optional): Flag to enable or disable TensorBoard logging.
                Default is `True`.

        """
        self.enabled = enabled
        if self.enabled:
            self.writer = SummaryWriter(
                log_dir=log_dir,
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix,
            )
            print(f"TensorBoard logging enabled. Logs will be saved to {self.writer.log_dir}")
        else:
            self.writer = None
            print("TensorBoard logging is disabled.")

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None) -> None:
        """
        Log a scalar value.

        Args:
            tag (str): Identifier for the scalar (e.g., "Loss/train").
            scalar_value (float): The scalar value to log.
            global_step (int, optional): Global step value to record with the scalar.

        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        global_step: int | None = None,
    ) -> None:
        """
        Log multiple scalar values under a main tag.

        Args:
            main_tag (str): The main tag under which the scalars are grouped (e.g., "Loss").
            tag_scalar_dict (Dict[str, float]): A dictionary of tag-scalar pairs to log.
                Example: {"train": 0.5, "test": 0.4}
            global_step (int, optional): Global step value to record with the scalars.

        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: int | None = None,
        bins: str = "tensorflow",
    ) -> None:
        """
        Log a histogram of tensor values.

        Args:
            tag (str): Identifier for the histogram (e.g., "weights/layer1").
            values (torch.Tensor or numpy.ndarray): Values to build the histogram.
            global_step (int, optional): Global step value to record with the histogram.
            bins (str, optional): Method for determining histogram bins. Default is 'tensorflow'.

        """
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, global_step, bins=bins)

    def add_image(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        global_step: int | None = None,
        dataformats: str = "CHW",
    ) -> None:
        """
        Log an image.

        Args:
            tag (str): Identifier for the image (e.g., "input_images").
            img_tensor (torch.Tensor or numpy.ndarray): Image data to log.
            global_step (int, optional): Global step value to record with the image.
            dataformats (str, optional): Format of the image data (e.g., 'CHW', 'HWC').
                Default is 'CHW'.

        """
        if self.enabled and self.writer is not None:
            self.writer.add_image(tag, img_tensor, global_step, dataformats=dataformats)

    def add_graph(
        self,
        model: torch.nn.Module,
        input_to_model: Any,
        verbose: bool = False,
    ) -> None:
        """
        Log the computational graph of the model.

        Args:
            model (torch.nn.Module): The PyTorch model to visualize.
            input_to_model (torch.Tensor or tuple of torch.Tensor): Input tensor(s) fed into
                the model.
            verbose (bool, optional): Whether to print the graph structure in the console.
                Default is `False`.

        """
        if self.enabled and self.writer is not None:
            self.writer.add_graph(model, input_to_model, verbose=verbose)

    def add_figure(
        self,
        tag: str,
        figure: plt.Figure,
        global_step: int | None = None,
        close: bool = True,
    ) -> None:
        """
        Log a matplotlib figure.

        Args:
            tag (str): Identifier for the figure (e.g., "precision_recall_curve").
            figure (matplotlib.pyplot.Figure): The matplotlib figure to log.
            global_step (int, optional): Global step value to record with the figure.
            close (bool, optional): Whether to close the figure after logging. Default is `True`.

        """
        if self.enabled and self.writer is not None:
            self.writer.add_figure(tag, figure, global_step, close)

    def add_class_accuracy(
        self,
        class_names: list[str],
        class_accuracy: list[float],
        global_step: int = 0,
        title: str = "Per-Class Accuracy",
    ) -> None:
        """
        Log per-class accuracy as a bar chart to TensorBoard.

        Args:
            class_names (List[str]): List of class names.
            class_accuracy (List[float]): Corresponding accuracies for each class.
            global_step (int, optional): Global step value to associate with the logged data.
            title (str, optional): Title of the bar chart.

        """
        if self.enabled and self.writer is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(class_names, class_accuracy, color="skyblue")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(title)
            ax.set_ylim(0, 100)  # Assuming accuracy is in percentage
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            self.writer.add_figure(title, fig, global_step)
            plt.close(fig)

    def add_text(
        self,
        tag: str,
        text: str,
        global_step: int | None = None,
    ) -> None:
        """
        Log a text string.

        Args:
            tag (str): Identifier for the text (e.g., "model_summary").
            text (str): The text string to log.
            global_step (int, optional): Global step value to record with the text.

        """
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, global_step)

    def flush(self):
        """
        Flushes the event file to disk.

        Ensures that all pending logs are written to disk. Useful for debugging or ensuring logs
        are saved at specific checkpoints.
        """
        if self.enabled and self.writer is not None:
            self.writer.flush()

    def close(self):
        """
        Close the TensorBoard SummaryWriter.

        It is important to close the writer to ensure all pending logs are flushed and resources
        are released.
        """
        if self.enabled and self.writer is not None:
            self.writer.close()
