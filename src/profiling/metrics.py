import csv
import time
from datetime import datetime

import psutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

DELAY = 1.00  # seconds
MB_UNIT = 2**20


def compute_maccs(model: Model, outfile) -> int:
    tf.keras.backend.clear_session()
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    graph_info = profile(
        forward_pass.get_concrete_function().graph,
        options=ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).with_file_output(outfile=outfile).build(),
    )
    # The //2 is necessary since `profile` counts multiply and accumulate
    # as two flops, here we report the total number of multiply accumulate ops
    maccs = graph_info.total_float_ops // 2
    print("MACCS: {:,}".format(maccs))
    with open(outfile, "a") as f:
        f.write(f"\nMACCS: {maccs:,}\n")
    return maccs


def get_basic_metrics(n_cpus, p):
    # Divide by the number of available CPUs to obtain the average CPU usage.
    return [datetime.now().isoformat(), round(p.cpu_percent() / n_cpus, 3), round(p.memory_info().rss / MB_UNIT, 3)]


def collect_linux_metrics(pid, out_file):
    n_cpus = psutil.cpu_count()
    p = psutil.Process(pid)
    p.cpu_percent()  # Call to avoid a 0% in the first registered entry
    with open(out_file, "w", newline="") as f:
        print("CPU and RAM profiling started...")
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cpu usage (%)", "memory usage (MB)", "temperature (Celsius)"])
        while True:
            with p.oneshot():
                data = get_basic_metrics(n_cpus, p)
                temps = psutil.sensors_temperatures()
                # Key for AMD CPUs in Linux
                if "k10temp" in temps.keys():
                    temp = temps["k10temp"][-1].current
                elif "coretemp" in temps.keys():
                    temp = temps["coretemp"][-1].current
                data.append(temp)
            writer.writerow(data)
            f.flush()
            time.sleep(DELAY)


def collect_windows_metrics(pid: int, out_file: str):
    n_cpus = psutil.cpu_count()
    p = psutil.Process(pid)
    p.cpu_percent()  # Call to avoid a 0% in the first registered entry
    with open(out_file, "w", newline="") as f:
        print("CPU and RAM profiling started...")
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cpu usage (%)", "memory usage (MB)"])
        while True:
            with p.oneshot():
                data = get_basic_metrics(n_cpus, p)
            writer.writerow(data)
            f.flush()
            time.sleep(DELAY)
