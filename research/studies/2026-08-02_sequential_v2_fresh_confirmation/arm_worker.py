"""One-shot entry point for exactly one Sequential policy arm."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import isolation_protocol as protocol
import run_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    try:
        job = json.loads(args.job.read_text(encoding="utf-8"))
        protocol.validate_job(job)
        protocol.assert_worker_environment()
        result = protocol.seal_result(run_study.execute_arm_job(job))
        protocol.validate_result_envelope(result, job)
        protocol.write_json_atomic(args.result, result)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
