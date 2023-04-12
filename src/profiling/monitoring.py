import multiprocessing as mp
import os
import sys

from src.models.train_model import run_main
from src.profiling.metrics import collect_linux_metrics, collect_windows_metrics


def monitor_training(arch, train_ds, train_size, val_ds, val_size, out_file):
    pid = os.getpid()
    arguments = (pid, out_file)
    if sys.platform == "linux":
        worker_process = mp.Process(target=collect_linux_metrics, args=arguments)
    elif sys.platform == "win32":
        worker_process = mp.Process(target=collect_windows_metrics, args=arguments)
    else:
        raise NotImplementedError("The monitoring function is not implemented for this OS.")
    worker_process.start()

    run_id = None
    acc_plot = None
    model = None
    try:
        run_id, acc_plot, model = run_main(arch, train_ds, train_size, val_ds, val_size, reproducible=True)
    except Exception as e:
        print(e)

    worker_process.terminate()

    return run_id, acc_plot, model
