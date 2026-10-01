import csv
import logging
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import GPUtil as GPU
import humanize
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

TD = TypeVar("TD", bound=dict)


class ExperimentHelper:
    """
    Namespace of static helpers for common experiment setup: shell commands,
    logging, device/environment info, reproducibility seeding, and interactive
    enum selection. Not meant to be instantiated.
    """

    @staticmethod
    def get_cmd_output(command: str) -> str:
        """
        Runs a shell command and returns its combined stdout/stderr output.

        Args:
            command (str): The shell command to run.

        Returns:
            str: The command's decoded output.
        """
        return subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            shell=True
        ).decode('UTF-8')

    @staticmethod
    def setup_logging(name: str, log_file: Path) -> logging.Logger:
        """
        Creates (or reconfigures) a logger that writes to both stdout and a log file.

        Args:
            name (str): Name of the logger, used to key it in the logging registry.
            log_file (Path): Path to the log file. Appended to across runs; the
                parent directory is created if missing.

        Returns:
            logging.Logger: The configured logger.
        """
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # avoid duplicate handlers if this is called again (e.g. cell re-run)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%d/%m/%Y %H:%M:%S"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

        return logger

    @staticmethod
    def start_experiment(name: str, base_dir: Path) -> tuple[str, Path, logging.Logger]:
        """
        Starts a new documented experiment run: generates a timestamp-based
        experiment id, creates its output directory under `base_dir`, and sets
        up a logger that writes into that directory.

        Args:
            name (str): Name of the experiment/model (e.g. 'random_forest'),
                used as both the experiment id prefix and the logger name.
            base_dir (Path): Directory under which the experiment's output
                directory is created (e.g. 'experiments/entries').

        Returns:
            tuple[str, Path, logging.Logger]: The experiment id, its output
                directory, and a logger writing to '{experiment_dir}/experiment.log'.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        experiment_dir = Path(base_dir) / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        logger = ExperimentHelper.setup_logging(name, experiment_dir / "experiment.log")
        logger.info(f"Started experiment '{experiment_id}'. Experiment start time: '{datetime.now(timezone.utc)}'")

        return experiment_id, experiment_dir, logger

    @staticmethod
    def get_device_info(logger: logging.Logger | None = None) -> dict:
        """
        Collects CPU, GPU, RAM, and CUDA information about the host machine.

        Args:
            logger (logging.Logger | None): If given, each collected value is
                logged at INFO level.

        Returns:
            dict: Mapping with keys 'cpu', 'physical_cpu_count',
                'logical_cpu_count', 'cuda_version', 'gpu', 'general_ram_gb',
                and 'gpu_ram_total_mb'.
        """
        cpu = ExperimentHelper.get_cmd_output('cat /proc/cpuinfo | grep -E "model name"')
        cpu = cpu.split('\n')[0].split('\t: ')[-1]
        physical_cpu_count = psutil.cpu_count(logical=False)
        logical_cpu_count = psutil.cpu_count()  # physical count X no. of threads per physical core
        try:
            cuda_version = ExperimentHelper.get_cmd_output('nvcc --version | grep -E "Build"')
        except subprocess.CalledProcessError:
            cuda_version = "Not Available"
        try:
            gpu = ExperimentHelper.get_cmd_output("nvidia-smi -L")
        except subprocess.CalledProcessError:
            gpu = "Not Available"
        general_ram_gb = humanize.naturalsize(psutil.virtual_memory().available)
        try:
            gpu_ram_total_mb = GPU.getGPUs()[0].memoryTotal
        except IndexError:
            gpu_ram_total_mb = "Not Available"

        device_info = {
            "cpu": cpu,
            "physical_cpu_count": physical_cpu_count,
            "logical_cpu_count": logical_cpu_count,
            "cuda_version": cuda_version,
            "gpu": gpu,
            "general_ram_gb": general_ram_gb,
            "gpu_ram_total_mb": gpu_ram_total_mb,
        }

        if logger is not None:
            logger.info(f"CPU: '{cpu}'")
            logger.info(f"Physical CPU Count: '{physical_cpu_count}'")
            logger.info(f"Logical CPU Count: '{logical_cpu_count}'")
            logger.info(f"CUDA Version: '{cuda_version}'")
            logger.info(f"GPU: '{gpu}'")
            logger.info(f"Available RAM: '{general_ram_gb}'")
            logger.info(f"GPU RAM: '{gpu_ram_total_mb}'")

        return device_info

    @staticmethod
    def set_global_seed(seed: int) -> torch.device:
        """
        Seeds torch, numpy, and the stdlib random module for reproducibility,
        and resolves the torch device to use.

        Args:
            seed (int): The seed to apply across all random number generators.

        Returns:
            torch.device: 'cuda' if a GPU is available, else 'cpu'.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False  # selects fastest conv algo
        torch.backends.cudnn.deterministic = True
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def save_typed_dict_to_csv(save_path: Path, dict_type: type[TD], value_list: list[TD]) -> None:
        """
        Writes a list of TypedDict records to a CSV file, using the TypedDict's
        fields (in declaration order) as the CSV columns.

        Args:
            save_path (Path): Path of the CSV file to write (overwritten if it exists).
            dict_type (type[TD]): The TypedDict class describing `value_list`'s
                records; its field names become the CSV header/columns.
            value_list (list[TD]): The records to write, one per row.
        """
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dict_type.__annotations__.keys())
            writer.writeheader()
            writer.writerows(value_list)


    @staticmethod
    def save_roc_plot(plots_dir: Path, run_name: str, round_idx: int, fpr, tpr, roc: float) -> None:
        """
        Plots and saves an ROC curve to '{plots_dir}/{run_name}_round{round_idx}_roc.png'.

        Args:
            plots_dir (Path): Directory to save the plot in. Created if missing.
            run_name (str): Name of the run, used in the plot title and filename.
            round_idx (int): Index of the round, used in the plot title and filename.
            fpr: False positive rates, as returned by sklearn's roc_curve.
            tpr: True positive rates, as returned by sklearn's roc_curve.
            roc (float): ROC AUC score, shown in the plot title.
        """
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="tab:blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {run_name} - Round {round_idx} (AUC = {roc:.3f})")
        fig.savefig(plots_dir / f"{run_name}_round{round_idx}_roc.png")
        plt.close(fig)

    @staticmethod
    def save_prc_plot(plots_dir: Path, run_name: str, round_idx: int, precision, recall, prc: float) -> None:
        """
        Plots and saves a precision-recall curve to '{plots_dir}/{run_name}_round{round_idx}_prc.png'.

        Args:
            plots_dir (Path): Directory to save the plot in. Created if missing.
            run_name (str): Name of the run, used in the plot title and filename.
            round_idx (int): Index of the round, used in the plot title and filename.
            precision: Precision values, as returned by sklearn's precision_recall_curve.
            recall: Recall values, as returned by sklearn's precision_recall_curve.
            prc (float): Average precision score, shown in the plot title.
        """
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="tab:orange")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PRC Curve - {run_name} - Round {round_idx} (AP = {prc:.3f})")
        fig.savefig(plots_dir / f"{run_name}_round{round_idx}_prc.png")
        plt.close(fig)
