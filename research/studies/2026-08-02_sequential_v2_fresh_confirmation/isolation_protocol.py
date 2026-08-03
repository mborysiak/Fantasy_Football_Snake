"""Standard-library protocol for one-shot Sequential policy-arm workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


JOB_SCHEMA_VERSION = "sequential-v2-policy-arm-job-v1"
RESULT_SCHEMA_VERSION = "sequential-v2-policy-arm-result-v1"
EXECUTION_ISOLATION = "fresh_subprocess_per_policy_arm"
WORKER_ENVIRONMENT = {
    "PYTHONHASHSEED": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_json_hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def seal_job(payload):
    job = dict(payload)
    job["schema_version"] = JOB_SCHEMA_VERSION
    job.pop("job_sha256", None)
    job["job_sha256"] = stable_json_hash(job)
    return job


def validate_job(job):
    if job.get("schema_version") != JOB_SCHEMA_VERSION:
        raise ValueError("Policy-arm job schema version is unsupported.")
    claimed = job.get("job_sha256")
    unsigned = dict(job)
    unsigned.pop("job_sha256", None)
    if not isinstance(claimed, str) or claimed != stable_json_hash(unsigned):
        raise ValueError("Policy-arm job hash is missing or invalid.")
    if job.get("arm") not in {"primary", "wide"}:
        raise ValueError("Policy-arm label must be primary or wide.")
    if int(job.get("candidate_count", 0)) <= 0:
        raise ValueError("Policy-arm candidate count must be positive.")
    return job


def worker_environment(base_environment=None):
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(WORKER_ENVIRONMENT)
    return environment


def observed_worker_environment():
    return {key: os.environ.get(key) for key in WORKER_ENVIRONMENT}


def assert_worker_environment():
    observed = observed_worker_environment()
    if observed != WORKER_ENVIRONMENT:
        raise RuntimeError(
            "Worker process did not start under the frozen hash/native-thread "
            f"environment: {observed!r}."
        )
    return observed


def _package_root(distribution):
    try:
        spec = importlib.util.find_spec(distribution)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0]).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def _native_binary_paths():
    """Return a deterministic, high-risk native-binary inventory.

    The selected extension modules cover each numerical runtime used by the
    Sequential worker. Vendor BLAS and CVXOPT/GLPK DLLs are included because
    they appeared in the invalidated Windows crash reports, even though the
    Sequential helper now lazy-loads CVXOPT/GLPK.
    """

    paths = {Path(sys.executable).resolve()}
    for directory in {Path(sys.base_prefix), Path(sys.prefix)}:
        paths.update(path.resolve() for path in directory.glob("python3*.dll"))

    patterns = {
        "numpy": (
            "_core/_multiarray_umath*.pyd",
            "core/_multiarray_umath*.pyd",
            "_core/_multiarray_tests*.pyd",
        ),
        "pandas": ("_libs/algos*.pyd", "_libs/hashtable*.pyd"),
        "scipy": ("_lib/_ccallback_c*.pyd",),
        "cvxopt": ("base*.pyd", "glpk*.pyd"),
    }
    for package, package_patterns in patterns.items():
        root = _package_root(package)
        if root is None:
            continue
        for pattern in package_patterns:
            paths.update(path.resolve() for path in root.glob(pattern))
        for vendor_dir in (
            root / ".libs",
            root / ".lib",
            root.parent / f"{package}.libs",
        ):
            if vendor_dir.is_dir():
                paths.update(path.resolve() for path in vendor_dir.glob("*.dll"))
    return sorted(path for path in paths if path.is_file())


def native_binary_fingerprint():
    entries = [
        {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in _native_binary_paths()
    ]
    return {
        "selection": "python_and_high_risk_numeric_native_artifacts_v1",
        "file_count": len(entries),
        "manifest_sha256": stable_json_hash(entries),
        "files": entries,
    }


def runtime_fingerprint():
    packages = {}
    for distribution in ("numpy", "pandas", "scipy", "scikit-learn", "cvxopt"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "worker_environment_contract": dict(WORKER_ENVIRONMENT),
        "native_binaries": native_binary_fingerprint(),
    }


def validate_result_envelope(result, job):
    validate_job(job)
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Policy-arm result schema version is unsupported.")
    if result.get("job_sha256") != job["job_sha256"]:
        raise ValueError("Policy-arm result belongs to a different job.")
    if result.get("arm") != job["arm"]:
        raise ValueError("Policy-arm result label differs from its job.")
    if int(result.get("candidate_count", -1)) != int(job["candidate_count"]):
        raise ValueError("Policy-arm result candidate count differs from its job.")
    if result.get("state") != job["state"]:
        raise ValueError("Policy-arm result state differs from its job.")
    if result.get("source_fingerprints") != job["source_fingerprints"]:
        raise ValueError("Policy-arm worker source fingerprints differ from its job.")
    if result.get("runtime_fingerprint") != job["runtime_fingerprint"]:
        raise ValueError("Policy-arm worker runtime/native fingerprint differs.")
    if result.get("observed_worker_environment") != WORKER_ENVIRONMENT:
        raise ValueError("Policy-arm worker environment attestation differs.")
    claimed = result.get("result_sha256")
    unsigned = dict(result)
    unsigned.pop("result_sha256", None)
    if not isinstance(claimed, str) or claimed != stable_json_hash(unsigned):
        raise ValueError("Policy-arm result hash is missing or invalid.")
    return result


def seal_result(payload):
    result = dict(payload)
    result["schema_version"] = RESULT_SCHEMA_VERSION
    result.pop("result_sha256", None)
    result["result_sha256"] = stable_json_hash(result)
    return result


def write_json_atomic(path, payload):
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
