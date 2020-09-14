import json
import sys
import copy
import argparse
import os
sys.path.append("../")
from driver_config import OLTPBENCH_HOME  # pylint: disable=import-error,wrong-import-position  # noqa: E402

parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
parser.add_argument("result_dir")
args = parser.parse_args()  # pylint: disable=invalid-name

INTEGER = 2

USER_DEFINED_METRICS = {
    "throughput": {
        "unit": "transaction / second",
        "short_unit": "txn/s",
        "type": INTEGER
    },
    "latency_99": {
        "unit": "milliseconds",
        "short_unit": "ms",
        "type": INTEGER
    },
    "latency_95": {
        "unit": "milliseconds",
        "short_unit": "ms",
        "type": INTEGER
    }
}


def get_udm():
    summary_path = OLTPBENCH_HOME + '/results/outputfile.summary'
    with open(summary_path, 'r') as f:
        info = json.load(f)
    metrics = copy.deepcopy(USER_DEFINED_METRICS)
    metrics["throughput"]["value"] = info["Throughput (requests/second)"]
    metrics["latency_99"]["value"] =\
        info["Latency Distribution"]["99th Percentile Latency (milliseconds)"]
    metrics["latency_95"]["value"] =\
        info["Latency Distribution"]["95th Percentile Latency (milliseconds)"]
    return metrics


def write_udm():
    metrics = get_udm()
    result_dir = args.result_dir
    path = os.path.join(result_dir, 'user_defined_metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    write_udm()
