import logging
import random
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path

import GPUtil as GPU
import humanize
import numpy as np
import psutil
import torch


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
        logger.info(f"Started experiment '{experiment_id}'")

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
    def prompt_enum_choice(logger: logging.Logger, enum_cls: type[Enum], label: str, supported: list | None = None):
        """
        Prompts the user to select a member of an enum by index or name, logging
        the available options and the final selection.

        Args:
            logger (logging.Logger): Logger used to announce options and the selection.
            enum_cls (type[Enum]): The enum class to choose a member from. Must
                have string `.value`s.
            label (str): Human-readable name for what is being selected (e.g.
                'dataset source'), used in prompts and log messages.
            supported (list | None): If given, restricts valid choices to this
                subset of members; other members are listed but rejected.

        Returns:
            Enum: The selected enum member.

        Raises:
            ValueError: If the selection is not a valid member of `enum_cls`,
                or is not in `supported` when given.
        """
        options = list(enum_cls)
        logger.info(f"Available {label}s:")
        for i, option in enumerate(options):
            note = "" if supported is None or option in supported else " (not supported)"
            logger.info(f"  {i}: {option.value}{note}")

        selection = input(f"Select {label} (name or index): ").strip()
        if selection.isdigit():
            choice = options[int(selection)]
        else:
            choice = enum_cls(selection.lower())

        if supported is not None and choice not in supported:
            raise ValueError(
                f"{label.capitalize()} '{choice.value}' is not supported here. "
                f"Supported {label}s: {[o.value for o in supported]}"
            )

        logger.info(f"Selected {label}: '{choice.value}'")
        return choice
