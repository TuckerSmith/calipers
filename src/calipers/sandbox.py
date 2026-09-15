"""Run generated build123d code in a subprocess and measure what it produced.

Contract for the generated script: it must leave a build123d ``Part``/``Solid``/``Compound`` (or a
CadQuery ``Workplane``) in a module-level variable named ``result``. The runner appends a footer that
exports ``result`` to STEP and STL, then loads the STEP with calipers, builds the geometry report,
optionally verifies it against a spec, and lints the source for dimension provenance.

Errors come back structured (exception type, message, the offending line) so a model can repair
its code instead of guessing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SANDBOX_MEMORY_MB = int(os.environ.get("CALIPERS_SANDBOX_MEMORY_MB", "8192"))  # address-space cap for the script

FOOTER = r"""

# ---- calipers runner footer (appended automatically) ----
import json as _json, sys as _sys
try:
    _res = result
except NameError:
    print(_json.dumps({"calipers_error": "the script must assign the finished shape to a variable named `result`"}))
    _sys.exit(3)
try:
    import cadquery as _cq  # optional interop
    if isinstance(_res, _cq.Assembly):
        _res = _res.toCompound()
    if isinstance(_res, _cq.Workplane):
        _res = _res.val()
    if hasattr(_res, "wrapped") and type(_res).__module__.startswith("cadquery"):
        from build123d import Compound as _Compound
        _res = _Compound.cast(_res.wrapped)
except ImportError:
    pass
from build123d import Shape as _Shape, export_step as _export_step, export_stl as _export_stl
if hasattr(_res, "part") and not isinstance(_res, _Shape):
    _res = _res.part
if not isinstance(_res, _Shape):
    print(_json.dumps({"calipers_error": f"`result` must be a build123d shape (Part/Solid/Compound), got {type(_res).__name__}"}))
    _sys.exit(3)
if len(_res.solids()) == 0:
    print(_json.dumps({"calipers_error": "`result` contains no solid (a sketch, face or wire cannot be verified as a part)"}))
    _sys.exit(3)
_export_step(_res, __OUT_STEP__)
_export_stl(_res, __OUT_STL__, tolerance=0.01, angular_tolerance=0.1)
print(_json.dumps({"calipers_exported": [__OUT_STEP__, __OUT_STL__]}))
"""


@dataclass
class RunResult:
    ok: bool
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    error: Optional[dict] = None
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0
    report_text: Optional[str] = None
    report: Optional[dict] = None
    verify: Optional[dict] = None
    verify_text: Optional[str] = None
    lint: Optional[dict] = None
    lint_text: Optional[str] = None
    renders: list[str] = field(default_factory=list)
    diff: Optional[dict] = None
    diff_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in {"report"}}

    @property
    def passed(self) -> bool:
        """Executed, lint clean and (when a spec was given) every contract passed."""
        return self.ok and (self.lint is None or bool(self.lint.get("ok", True))) and (self.verify is None or bool(self.verify["passed"]))

    def score(self) -> tuple:
        """Sortable quality: lower is better. Executed < lint clean < contracts passed < fewest failed
        checks < smallest total normalised deviation. Used by :func:`run_candidates` as the judge."""
        n_failed = self.verify["n_failed"] if self.verify else 0
        dev = 0.0
        for c in (self.verify or {}).get("checks", []):
            if not c["passed"] and c.get("deviation") is not None:
                dev += abs(float(c["deviation"]))
        lint_bad = 0 if self.lint is None else (len(self.lint.get("naked_numbers", [])) + len(self.lint.get("problems", [])))
        return (0 if self.ok else 1, 0 if self.verify is None or self.verify["passed"] else 1, n_failed, lint_bad, round(dev, 4))

    def to_text(self) -> str:
        parts = []
        if not self.ok:
            e = self.error or {}
            parts.append(f"# Execution FAILED: {e.get('type')}: {e.get('message')}")
            if e.get("line"):
                parts.append(f"at line {e['line']}: {e.get('source_line', '')}")
            if e.get("traceback_tail"):
                parts.append(e["traceback_tail"])
        else:
            parts.append(f"# Execution OK in {self.elapsed_s:.1f}s → {self.step_path}")
        if self.lint_text:
            parts.append(self.lint_text)
        if self.report_text:
            parts.append(self.report_text)
        if self.verify_text:
            parts.append(self.verify_text)
        if self.diff_text:
            parts.append(self.diff_text)
        return "\n\n".join(parts)


def _parse_error(stderr: str, code_lines: list[str], returncode: int | None = None, header_lines: int = 0) -> dict:
    tail = stderr.strip().splitlines()[-40:]
    if not tail:
        msg = "the script exited before the runner footer could export `result`" if returncode == 0 else f"exit code {returncode} with no traceback"
        return {"type": "ScriptExited", "message": msg, "line": None, "source_line": "", "traceback_tail": ""}
    # the exception line is the last one that looks like `SomeError: message`; the message may span lines
    exc_idx = max((i for i, t in enumerate(tail) if re.match(r"^[\w\.]*(Error|Exception|Warning|Exit)\b", t)), default=len(tail) - 1)
    m = re.match(r"([\w\.]+)\s*:\s*(.*)", tail[exc_idx])
    etype, msg = (m.group(1), " ".join([m.group(2)] + tail[exc_idx + 1 :]).strip()) if m else ("Error", " ".join(tail[exc_idx:]))
    line_no = None
    for t in reversed(tail[: exc_idx + 1]):
        mm = re.search(r"script\.py\", line (\d+)", t)
        if mm:
            line_no = int(mm.group(1)) - header_lines
            break
    note = ""
    if line_no is not None and line_no > len(code_lines):
        note = " (in the runner footer: `result` is probably not a valid shape)"
        line_no = None
    src = code_lines[line_no - 1].strip() if line_no and 0 < line_no <= len(code_lines) else ""
    return {"type": etype, "message": msg + note, "line": line_no, "source_line": src, "traceback_tail": "\n".join(tail[-12:])}


def run_code(
    code: str,
    workdir: str | os.PathLike | None = None,
    spec: Optional[dict] = None,
    timeout: float = 180.0,
    lint: bool = True,
    render: bool = False,
    sections: tuple[str, ...] = (),
    references: Optional[dict] = None,
    diff_previous: bool = True,
) -> RunResult:
    """Execute build123d code, then report / verify / lint. Never raises for script errors.

    When the spec names references, the script sees ``REFERENCES = {id: absolute path}`` (one header
    line) and ``measured:<id>...`` sources are checked against the loaded reference. When ``workdir``
    already holds a ``result.step`` from an earlier run, the new result is diffed against it.
    """
    import shutil
    import time

    from calipers.provenance import lint_source

    wd = (Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="calipers_run_"))).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    step, stl, script = wd / "result.step", wd / "result.stl", wd / "script.py"
    previous = wd / "previous.step"
    if diff_previous and step.exists():
        shutil.copyfile(step, previous)
    refs: dict = dict(references or {})
    header = ""
    if spec is not None and spec.get("references"):
        from calipers.reference import load_reference

        paths = {}
        for rf in spec["references"]:
            p = Path(rf["path"])
            if not p.is_absolute() and spec.get("_base_dir"):
                p = Path(spec["_base_dir"]) / p
            paths[rf["id"]] = str(p.resolve())
            if rf["id"] not in refs:
                refs[rf["id"]] = load_reference(rf, spec.get("_base_dir"))
        header = f"REFERENCES = {json.dumps(paths)}\n"
    header_lines = header.count("\n")
    footer = FOOTER.replace("__OUT_STEP__", json.dumps(str(step))).replace("__OUT_STL__", json.dumps(str(stl)))
    script.write_text(header + code + footer, encoding="utf-8")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-s", "-B", str(script)],
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "MPLBACKEND": "Agg"},
            preexec_fn=_limit_resources if os.name == "posix" else None,
        )
        stdout, stderr, code_ = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        return RunResult(False, error={"type": "Timeout", "message": f"script exceeded {timeout}s"}, stdout=exc.stdout or "", stderr=exc.stderr or "", elapsed_s=time.time() - t0)
    elapsed = time.time() - t0
    res = RunResult(ok=code_ == 0 and step.exists(), stdout=stdout[-4000:], stderr=stderr[-4000:], elapsed_s=elapsed)
    if lint:
        try:
            lr = lint_source(code)
            if spec is not None:
                from calipers.provenance import check_sources

                check_sources(lr, spec, refs)
            res.lint, res.lint_text = lr.to_dict(), lr.to_text()
        except SyntaxError as exc:
            res.lint = {"ok": False, "problems": [f"syntax error: {exc}"]}
            res.lint_text = f"# Provenance lint: could not parse ({exc})"
    if not res.ok:
        if "calipers_error" in stdout:
            try:
                msg = json.loads([ln for ln in stdout.splitlines() if "calipers_error" in ln][-1])["calipers_error"]
                res.error = {"type": "ContractError", "message": msg}
            except Exception:
                res.error = {"type": "ContractError", "message": stdout[-500:]}
        else:
            res.error = _parse_error(stderr, code.splitlines(), code_, header_lines)
        return res
    res.step_path, res.stl_path = str(step), str(stl)
    from calipers.model import Model
    from calipers.report import build_report

    model = Model.load(step)
    rep = build_report(model, sections=sections, symmetry=bool(spec and spec.get("symmetry")))
    res.report, res.report_text = rep.to_dict(), rep.to_text()
    if spec is not None:
        from calipers.contracts import verify

        vr = verify(model, spec, features=rep.features, references=refs)
        res.verify, res.verify_text = vr.to_dict(), vr.to_text()
    if diff_previous and previous.exists():
        from calipers.diff import diff_models, diff_text

        try:
            prev = Model.load(previous)
            prev.name, model.name = "previous run", "this run"
            d = diff_models(prev, model, features_b=rep.features)
            res.diff, res.diff_text = d, diff_text(d)
        except Exception as exc:  # a diff must never break a run
            res.diff_text = f"# Diff against the previous run failed: {exc}"
    if render:
        from calipers.render import render_views

        res.renders = render_views(model, wd / "renders", views=("iso", "front", "top"))
    return res


def _limit_resources() -> None:  # pragma: no cover - runs in the child
    """Cap the script's address space and file sizes. This is a *resource* limit, not a security
    boundary: generated code runs with the caller's privileges (see the sandbox docstring)."""
    try:
        import resource

        cap = SANDBOX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**3, 2 * 1024**3))
    except Exception:
        pass


def run_candidates(codes: list[str], spec: Optional[dict] = None, workdir: str | os.PathLike | None = None, workers: int = 4, **kw) -> list[tuple[int, RunResult]]:
    """Best-of-N: run every candidate script (in parallel subprocesses), verify each against the spec,
    and return ``(index, result)`` pairs sorted best first by :meth:`RunResult.score`. The verifier is
    the judge; the generator that produced the candidates is expected to be wrong in places."""
    from concurrent.futures import ThreadPoolExecutor

    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="calipers_best_"))
    kw.setdefault("diff_previous", False)

    def one(i: int) -> tuple[int, RunResult]:
        return i, run_code(codes[i], workdir=base / f"candidate_{i}", spec=spec, **kw)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(codes)))) as ex:
        results = list(ex.map(one, range(len(codes))))
    results.sort(key=lambda t: (t[1].score(), t[0]))
    return results


def candidates_text(ranked: list[tuple[int, RunResult]]) -> str:
    lines = [f"# Best-of-{len(ranked)}: candidate {ranked[0][0]} is best" + (" and passes everything" if ranked[0][1].passed else " but does not pass yet")]
    for i, res in ranked:
        sc = res.score()
        if not res.ok:
            what = f"execution failed: {(res.error or {}).get('type')}: {(res.error or {}).get('message', '')[:80]}"
        else:
            v = res.verify
            what = (f"{v['n_checks'] - v['n_failed']}/{v['n_checks']} contracts" if v else "no spec") + f", {sc[3]} lint problem(s), total deviation {sc[4]}"
        lines.append(f"- candidate {i}: {what} → {res.step_path or '-'}")
    return "\n".join(lines)


def api_help(name: str, max_lines: int = 60) -> str:
    """Signature and docstring of a build123d (or cadquery) object, or a name search for a fragment."""
    import inspect

    import build123d

    if hasattr(build123d, name):
        obj = getattr(build123d, name)
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = ""
        doc = inspect.getdoc(obj) or "(no docstring)"
        lines = doc.splitlines()[:max_lines]
        return textwrap.dedent(f"build123d.{name}{sig}\n\n" + "\n".join(lines))
    frag = name.lower()
    matches = sorted(n for n in dir(build123d) if frag in n.lower() and not n.startswith("_"))
    if not matches:
        return f"no build123d name contains {name!r}"
    return "Matching build123d names:\n" + "\n".join(matches[:80])
