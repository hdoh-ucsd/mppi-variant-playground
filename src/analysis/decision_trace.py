"""Versioned five-stage decision records, immutable configs and lossless NPZ arrays.

This module has no controller, plotting or simulation dependencies. Array references
are deduplicated within a decision; configurations are stored once per immutable hash.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
MODES = {"FROZEN_DECISION", "CLOSED_LOOP"}
STAGES = (
    "dynamics_task",
    "distributional_objective",
    "sample_refine",
    "weight_update",
    "feedback_execute",
)
STATUSES = {"available", "missing", "not_applicable", "mechanism_inactive", "numerical_failure"}
STATE_UNITS = ["m", "m", "rad", "rad", "m/s", "m/s", "rad/s"]
CONTROL_UNITS = ["rad", "normalized_signed_throttle"]


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def canonical(value):
    return json.dumps(native(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def problem_config(config):
    """Recording settings are not part of the optimization problem's identity."""
    c = json.loads(canonical(config))
    for key in ["saved_rollouts", "snapshot_steps"]:
        c.get("controller", {}).pop(key, None)
    for key in ["extra_snapshot_seeds", "snapshot_steps", "representative_seed"]:
        c.get("execution", {}).pop(key, None)
    return c


def missing(reason):
    return {"status": "missing", "reason": reason}


def not_applicable(reason):
    return {"status": "not_applicable", "reason": reason}


@dataclass
class StageRecord:
    status: str
    data: dict = field(default_factory=dict)
    missing_fields: list = field(default_factory=list)
    notes: list = field(default_factory=list)


@dataclass
class DecisionTrace:
    metadata: dict
    stages: dict
    arrays: dict = field(default_factory=dict)

    def put(self, value, units="dimensionless"):
        array = np.asarray(value)
        if array.dtype.kind not in "bfiu":
            raise ValueError("Only numerical arrays are allowed")
        array = np.ascontiguousarray(array)
        key = hashlib.sha256(
            str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
        ).hexdigest()
        self.arrays.setdefault(key, array.copy())
        return {"array": key, "shape": list(array.shape), "dtype": str(array.dtype), "units": units}

    def get(self, value):
        if isinstance(value, dict) and "array" in value:
            return self.arrays[value["array"]]
        if isinstance(value, dict) and "status" in value:
            return None
        return value

    @property
    def decision_id(self):
        return self.metadata["decision_id"]

    def document(self):
        return {
            "metadata": native(self.metadata),
            "stages": {key: native(vars(stage)) for key, stage in self.stages.items()},
        }

    def validate(self):
        m = self.metadata
        required = (
            "schema_version",
            "experiment_id",
            "scenario",
            "algorithm",
            "seed",
            "planning_step",
            "source_commit",
            "configuration_hash",
            "state_dimension",
            "control_dimension",
            "horizon",
            "sample_budget",
            "backend",
            "mode",
            "decision_id",
            "input_hash",
            "state_units",
            "control_units",
        )
        if any(key not in m for key in required):
            raise ValueError("Missing required DecisionTrace metadata")
        if m["schema_version"] != SCHEMA_VERSION or m["mode"] not in MODES:
            raise ValueError("Unsupported trace schema version or mode")
        if m["state_dimension"] != 7 or m["control_dimension"] != 2 or m["horizon"] < 1:
            raise ValueError("DecisionTrace requires seven-state/two-control dimensions")
        if m["state_units"] != STATE_UNITS or m["control_units"] != CONTROL_UNITS:
            raise ValueError("Incorrect state/control units")
        if list(self.stages) != list(STAGES):
            raise ValueError("Five ordered stages required")
        if any(s.status not in STATUSES for s in self.stages.values()):
            raise ValueError("Unknown stage classification")
        if m["decision_id"] != decision_id(m):
            raise ValueError("Unstable decision ID")

        def walk(value, label=""):
            if isinstance(value, dict):
                if "array" in value:
                    a = self.get(value)
                    if list(a.shape) != value["shape"] or str(a.dtype) != value["dtype"]:
                        raise ValueError("Array descriptor mismatch")
                    if not value.get("units"):
                        raise ValueError("Array units required")
                    if np.isnan(a).any() or np.isposinf(a).any():
                        raise ValueError("Invalid numerical array")
                    if np.isneginf(a).any() and "log_" not in label:
                        raise ValueError("Negative infinity only allowed in log scores")
                else:
                    for k, v in value.items():
                        walk(v, k)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    walk(v, label)

        walk(self.document())
        h = m["horizon"]
        state = self.get(self.stages["dynamics_task"].data.get("observed_state"))
        if state is not None and np.shape(state) != (7,):
            raise ValueError("Observed state dimensions")
        populations = self.stages["sample_refine"].data.get("populations", [])
        ids = [p["population_id"] for p in populations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate population ID")
        for p in populations:
            states = self.get(p["states"])
            latent = self.get(p["latent_controls"])
            controls = self.get(p["physical_controls"])
            k = p["samples"]
            for value, shape in [
                (states, (k, h + 1, 7)),
                (latent, (k, h, 2)),
                (controls, (k, h, 2)),
            ]:
                if value is not None and value.shape != shape:
                    raise ValueError("Population dimensions")
        updates = self.stages["weight_update"].data.get("updates", [])
        for u in updates:
            if u["population_id"] not in ids:
                raise ValueError("Update without originating population")
            w = self.get(u["weights"])
            costs = self.get(u["raw_costs"])
            k = next(p["samples"] for p in populations if p["population_id"] == u["population_id"])
            if w is not None:
                if w.shape != (k,) or (w < 0).any() or not np.isclose(w.sum(), 1, atol=1e-12):
                    raise ValueError("Weights must be normalized")
                if not np.isclose(u["ess"], 1 / np.sum(w * w)):
                    raise ValueError("Incorrect optimizer ESS")
                if costs is not None and costs.shape != w.shape:
                    raise ValueError("Cost/weight dimensions")
            plan = self.get(u["updated_latent"])
            if plan is not None and plan.shape != (h, 2):
                raise ValueError("Updated control dimensions")
        if m["algorithm"] == "dial_mpc" and populations:
            if [p["iteration"] for p in populations] != list(range(len(populations))):
                raise ValueError("DIAL iteration ordering")
        cc = self.stages["sample_refine"].data.get("cc", {})
        covariance = self.get(cc.get("predicted_covariances"))
        if covariance is not None:
            if covariance.shape != (h + 1, 7, 7):
                raise ValueError("Covariance dimensions")
            if not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-10):
                raise ValueError("Covariance must be symmetric")
            if np.linalg.eigvalsh(covariance).min() < -1e-9:
                raise ValueError("Covariance must be PSD")
        execution = self.stages["feedback_execute"].data
        if "decision_id" in execution and execution["decision_id"] != self.decision_id:
            raise ValueError("Execution decision ID mismatch")
        canonical(self.document())
        return self


def decision_id(metadata):
    identity = {
        k: metadata[k]
        for k in (
            "schema_version",
            "experiment_id",
            "scenario",
            "algorithm",
            "seed",
            "planning_step",
            "configuration_hash",
            "mode",
            "input_hash",
        )
    }
    return "decision-" + fingerprint(identity)[:24]


def new_trace(
    config, *, experiment_id, seed, planning_step, mode, inputs, source_commit, provenance=None
):
    c = problem_config(config)
    m = dict(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        scenario=c["scenario"]["name"],
        algorithm=c["controller"]["algorithm"],
        seed=int(seed),
        planning_step=int(planning_step),
        source_commit=source_commit,
        configuration_hash=fingerprint(c),
        state_dimension=7,
        control_dimension=2,
        horizon=c["controller"]["horizon"],
        sample_budget=c["controller"]["samples"],
        sample_budget_semantics="Candidates per population/pass; see stage 3 for total work",
        backend={"numpy": np.__version__, "floating_point": "float64", "rng": "PCG64"},
        mode=mode,
        input_hash=fingerprint(inputs),
        provenance=provenance or {},
        state_units=STATE_UNITS,
        control_units=CONTROL_UNITS,
    )
    m["decision_id"] = decision_id(m)
    return DecisionTrace(m, {key: StageRecord("missing") for key in STAGES})


class TraceStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "configurations").mkdir(exist_ok=True)

    def configuration(self, config):
        c = problem_config(config)
        key = fingerprint(c)
        path = self.root / "configurations" / f"{key}.json"
        if path.exists():
            if fingerprint(json.loads(path.read_text())) != key:
                raise ValueError("Immutable configuration hash mismatch")
        else:
            path.write_text(canonical(c) + "\n")
        return key

    def save(self, trace, config, *, reuse_sources=()):
        trace.validate()
        if self.configuration(config) != trace.metadata["configuration_hash"]:
            raise ValueError("Trace/configuration mismatch")
        folder = self.root / trace.decision_id
        folder.mkdir(exist_ok=False)
        external = {}
        for source in reuse_sources:
            source = Path(source)
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            with np.load(source, allow_pickle=False) as archive:
                for key in archive.files:
                    candidates = [(None, archive[key])]
                    if key.startswith("inner_"):
                        candidates += list(enumerate(archive[key]))
                    for index, value in candidates:
                        if value.dtype.kind not in "bfiu":
                            continue
                        value = np.ascontiguousarray(value)
                        identity = hashlib.sha256(
                            str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
                        ).hexdigest()
                        if identity in trace.arrays:
                            external[identity] = dict(
                                path=os.path.relpath(source, folder),
                                key=key,
                                index=index,
                                sha256=checksum,
                            )
        with (folder / "arrays.npz").open("xb") as f:
            np.savez_compressed(
                f,
                schema_version=np.array(SCHEMA_VERSION),
                **{k: v for k, v in trace.arrays.items() if k not in external},
            )
        document = trace.document()
        document["external_arrays"] = external
        document["arrays_sha256"] = hashlib.sha256((folder / "arrays.npz").read_bytes()).hexdigest()
        (folder / "trace.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        return folder

    def load(self, identifier):
        folder = self.root / identifier
        document = json.loads((folder / "trace.json").read_text())
        if (
            hashlib.sha256((folder / "arrays.npz").read_bytes()).hexdigest()
            != document["arrays_sha256"]
        ):
            raise ValueError("Array archive hash mismatch")
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            if str(archive["schema_version"]) != SCHEMA_VERSION:
                raise ValueError("Unsupported array schema version")
            arrays = {k: archive[k] for k in archive.files if k != "schema_version"}
        sources = {}
        for key, ref in document.get("external_arrays", {}).items():
            path = folder / ref["path"]
            if ref["path"] not in sources:
                if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                    raise ValueError("External array source hash mismatch")
                with np.load(path, allow_pickle=False) as archive:
                    sources[ref["path"]] = {k: archive[k] for k in archive.files}
            value = sources[ref["path"]][ref["key"]]
            arrays[key] = value if ref["index"] is None else value[ref["index"]]
        trace = DecisionTrace(
            document["metadata"], {k: StageRecord(**document["stages"][k]) for k in STAGES}, arrays
        )
        path = self.root / "configurations" / f"{trace.metadata['configuration_hash']}.json"
        if fingerprint(json.loads(path.read_text())) != trace.metadata["configuration_hash"]:
            raise ValueError("Configuration hash mismatch")
        return trace.validate()
