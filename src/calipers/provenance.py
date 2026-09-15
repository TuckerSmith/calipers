"""Dimension provenance: "no naked numbers" in generated CAD code.

The rule: every numeric literal that shapes geometry must live in a module-level ``PARAMS`` dict,
paired with a *source* string saying where it came from::

    PARAMS = {
        "plate_l": (60.0, "spec:envelope.extents.x"),
        "hole_d":  (5.0,  "spec:mount_holes.diameter"),
        "wall":    (2.4,  "assumption:3 perimeters of a 0.4 mm nozzle"),
        "boss_r":  (6.0,  "derived:boss_d/2"),
        "ref_w":   (31.004, "measured:3DBenchy.stl#envelope.y"),
    }

Sources must start with one of ``spec:``, ``measured:``, ``derived:``, ``standard:`` or
``assumption:``. Assumptions are allowed but reported separately so a reviewer sees every number
the model invented. Anywhere else in the code, numeric literals are *naked* — except a small set
of structural constants (0, 1, -1, 2, 0.5 and 90/180/360 for angles) and integers used as counts
or indices (``range``, subscripts, ``count=``, ``side_count=``).

This is a discipline check with a reviewer behind it, not a security boundary: a determined author
can still smuggle a number (``range(61)[-1]``); the point is that honest code cannot *accidentally*
carry an unsourced dimension, and every remaining number is visible in the assumptions list.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

SOURCE_PREFIXES = ("spec:", "measured:", "derived:", "standard:", "assumption:")
STRUCTURAL = {0, 1, -1, 2, 0.5, 90, 180, 360}
COUNT_KEYWORDS = {"count", "side_count", "n", "num", "steps", "segments", "copies", "rows", "cols", "columns"}


@dataclass
class Naked:
    value: float
    line: int
    col: int
    context: str


@dataclass
class LintResult:
    params: dict = field(default_factory=dict)  # name → (value_repr, source)
    naked: list[Naked] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # PARAMS entries with missing/invalid sources
    assumptions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.naked and not self.problems

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "params": self.params,
            "naked_numbers": [{"value": n.value, "line": n.line, "col": n.col, "context": n.context} for n in self.naked],
            "problems": self.problems,
            "assumptions": self.assumptions,
        }

    def to_text(self) -> str:
        lines = [f"# Provenance lint: {'PASS' if self.ok else 'FAIL'} — {len(self.params)} parameters, {len(self.naked)} naked numbers, {len(self.problems)} problems"]
        for p in self.problems:
            lines.append(f"[FAIL] {p}")
        for n in self.naked:
            lines.append(f"[FAIL] naked number {n.value!r} at line {n.line}:{n.col} in {n.context}")
        if self.assumptions:
            lines.append("Assumptions (numbers the author invented — review them):")
            lines += [f"  - {a}" for a in self.assumptions]
        return "\n".join(lines)


def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, complex):
        return abs(node.value)  # a complex literal is a number in disguise
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, source_lines: list[str]):
        self.lines = source_lines
        self.result = LintResult()
        self._skip: set[int] = set()  # ids of nodes whose literals are allowed
        self._params_seen = False

    # -- PARAMS dict
    def visit_Assign(self, node: ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "PARAMS" in names:
            if not isinstance(node.value, ast.Dict):
                self.result.problems.append(f"line {node.lineno}: PARAMS must be a literal dict, not {ast.unparse(node.value)[:40]!r}")
                self.generic_visit(node)
                return
            if self._params_seen:
                self.result.problems.append(f"line {node.lineno}: PARAMS is assigned more than once")
            self._params_seen = True
            self._collect_params(node.value)
            self._mark_allowed(node.value)
            return
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name) and node.target.id == "PARAMS":
            self.result.problems.append(f"line {node.lineno}: PARAMS must not be modified after its definition")
        self.generic_visit(node)

    def _collect_params(self, d: ast.Dict):
        for k, v in zip(d.keys, d.values):
            name = k.value if isinstance(k, ast.Constant) else ast.unparse(k)
            if isinstance(v, ast.Tuple) and len(v.elts) == 2:
                val, src = v.elts
                src_val = src.value if isinstance(src, ast.Constant) and isinstance(src.value, str) else None
                if src_val is None:
                    self.result.problems.append(f"PARAMS[{name!r}]: source must be a string literal")
                elif not src_val.startswith(SOURCE_PREFIXES):
                    self.result.problems.append(f"PARAMS[{name!r}]: source {src_val!r} must start with one of {SOURCE_PREFIXES}")
                elif not src_val.split(":", 1)[1].strip():
                    self.result.problems.append(f"PARAMS[{name!r}]: source {src_val!r} has no body — say which field, measurement, rule or assumption")
                else:
                    if src_val.startswith("assumption:"):
                        self.result.assumptions.append(f"{name} = {ast.unparse(val)} — {src_val[len('assumption:'):].strip()}")
                self.result.params[name] = (ast.unparse(val), src_val)
            else:
                self.result.problems.append(f"PARAMS[{name!r}]: value must be a (value, \"source\") tuple")

    def _mark_allowed(self, node: ast.AST):
        for n in ast.walk(node):
            self._skip.add(id(n))

    # -- allowed structural contexts
    def visit_Subscript(self, node: ast.Subscript):
        self._mark_allowed(node.slice)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        fname = ast.unparse(node.func)
        if fname in {"range", "enumerate", "len"}:  # counts: non-negative ints only
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, int) and not isinstance(a.value, bool) and a.value >= 0:
                    self._mark_allowed(a)
        if fname in {"float", "int", "eval", "Decimal", "Fraction", "complex"}:  # numbers smuggled in as strings
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    try:
                        self._report(a, float(a.value))
                    except ValueError:
                        pass
        for kw in node.keywords:
            v = kw.value
            if kw.arg in COUNT_KEYWORDS and isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool) and v.value >= 0:
                self._mark_allowed(v)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        # constant-fold expressions made only of literals: 2*2*2 is 8, not "structural"
        if id(node) in self._skip:
            return
        if all(isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Constant, ast.operator, ast.unaryop)) for n in ast.walk(node)):
            try:
                val = eval(compile(ast.Expression(body=node), "<fold>", "eval"), {"__builtins__": {}}, {})  # noqa: S307 - literals only
            except Exception:
                val = None
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if val not in STRUCTURAL:
                    self._report(node, val)
                return
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if id(node) in self._skip:
            return
        v = _literal_value(node)
        if v is None or v in STRUCTURAL:
            return
        self._report(node, v)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        v = _literal_value(node)
        if v is not None:
            if id(node) in self._skip or id(node.operand) in self._skip or v in STRUCTURAL:
                return
            self._report(node, v)
            return
        self.generic_visit(node)

    def _report(self, node: ast.AST, v):
        line = self.lines[node.lineno - 1].strip() if 0 < node.lineno <= len(self.lines) else ""
        self.result.naked.append(Naked(value=v, line=node.lineno, col=node.col_offset, context=line[:100]))


def lint_source(code: str) -> LintResult:
    tree = ast.parse(code)
    v = _Visitor(code.splitlines())
    v.visit(tree)
    if "PARAMS" not in {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}:
        v.result.problems.insert(0, "no module-level PARAMS dict: every dimension must be declared with a source")
    return v.result


def lint_file(path: str) -> LintResult:
    with open(path, encoding="utf-8") as f:
        return lint_source(f.read())


MEASURED_TOL = 0.05  # mm: a cited measurement may be rounded, not changed


def check_sources(result: LintResult, spec: dict, references: dict | None = None) -> list[str]:
    """Resolve every ``spec:`` source in a lint result against a normalized spec, and every
    ``measured:<reference id>.<path>`` source against the loaded reference model.

    Returns problems: sources that name nothing in the spec, and literal PARAMS values that disagree
    with the nominal the source points at (outside its tolerance). Appended to ``result.problems``.
    """
    import ast as _ast

    problems: list[str] = []
    ref_ids = {rf["id"] for rf in spec.get("references", [])}
    ref_cache: dict[str, dict] = {}
    for name, (value_repr, src) in result.params.items():
        if src and src.startswith("measured:") and ref_ids:
            path = src[len("measured:"):].strip()
            rid, _, rest = path.replace("#", ".").partition(".")
            if rid not in ref_ids:
                continue  # a free-form measurement note (e.g. from calipers on the bench)
            target = _resolve_measured(rid, rest, spec, references or {}, ref_cache)
            if target is _MISSING:
                problems.append(f"PARAMS[{name!r}]: source {src!r} names nothing measurable on reference {rid!r} (use bbox_min/bbox_max/extents/center.<x|y|z>, volume, or a feature id like C03.diameter)")
                continue
            try:
                lit = _ast.literal_eval(value_repr)
            except Exception:
                continue
            if isinstance(lit, (int, float)) and isinstance(target, (int, float)) and abs(float(lit) - float(target)) > MEASURED_TOL:
                problems.append(f"PARAMS[{name!r}] = {lit} but {src} measures {target} (differs by {abs(float(lit) - float(target)):.4f}; cite the measurement, do not change it)")
            continue
        if not src or not src.startswith("spec:"):
            continue
        path = src[len("spec:"):].strip()
        target = _resolve_spec_path(spec, path)
        if target is _MISSING:
            hint = " (an expression of spec values is a derived: source)" if any(ch in path for ch in "+-*/()") else ""
            problems.append(f"PARAMS[{name!r}]: source {src!r} does not exist in the spec{hint}")
            continue
        try:
            lit = _ast.literal_eval(value_repr)
        except Exception:
            continue  # a derived expression; nothing to compare
        if isinstance(target, tuple) and len(target) == 2 and isinstance(lit, (int, float)):
            nom, tol = target
            if abs(float(lit) - nom) > tol + 1e-9:
                problems.append(f"PARAMS[{name!r}] = {lit} but {src} is {nom} ± {tol}")
    result.problems.extend(problems)
    return problems


_MISSING = object()


def _resolve_measured(rid: str, path: str, spec: dict, references: dict, cache: dict):
    """``bbox_max.z`` / ``extents.x`` / ``center.y`` / ``volume`` / ``C03.diameter`` on a reference model."""
    from calipers.reference import load_reference, reference_summary

    if rid not in cache:
        model = references.get(rid)
        if model is None:
            rf = next(x for x in spec["references"] if x["id"] == rid)
            model = load_reference(rf, spec.get("_base_dir"))
            references[rid] = model
        cache[rid] = {"model": model, "summary": reference_summary(model), "features": None}
    entry = cache[rid]
    parts = [p for p in path.replace("[", ".").replace("]", "").split(".") if p]
    if not parts:
        return _MISSING
    node = entry["summary"]
    if parts[0] not in node:  # a feature id → the reference's feature report (computed once)
        if entry["features"] is None:
            from calipers.features import extract_features

            f = extract_features(entry["model"])
            entry["features"] = {e["id"]: e for e in f["cylinders"] + f["planes"] + f.get("slots", [])}
        node = entry["features"]
    for key in parts:
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, (list, tuple)) and key in ("x", "y", "z"):
            node = node["xyz".index(key)]
        elif isinstance(node, (list, tuple)) and key.lstrip("-").isdigit() and -len(node) <= int(key) < len(node):
            node = node[int(key)]
        else:
            return _MISSING
    return node


def _resolve_spec_path(spec: dict, path: str):
    """``envelope.extents.x`` / ``<feature>.diameter`` / ``<feature>.positions.1.0`` / ``planes.top.offset`` ..."""
    if any(ch in path for ch in "+-*/() ") and "[*]" not in path:
        return _MISSING  # an expression: that is a derived: source, not a spec field
    parts = [p for p in path.replace("[*]", ".*").replace("[", ".").replace("]", "").split(".") if p]
    if not parts:
        return _MISSING
    feats = {f["id"]: f for f in spec.get("features", [])}
    planes = {p["id"]: p for p in spec.get("planes", [])}
    node = spec
    if parts[0] == "features" and len(parts) > 1 and parts[1] in feats:
        parts = parts[1:]
    if parts[0] in feats:
        node = feats[parts[0]]
        parts = parts[1:]
        if parts and parts[0] in {"height", "depth"}:
            parts[0] = "length"
    elif parts[0] == "planes" and len(parts) > 1 and parts[1] in planes:
        node = planes[parts[1]]
        parts = parts[2:]
    for key in parts:
        if key == "*" and isinstance(node, (list, tuple)):
            return node  # "any instance": the list itself; no single nominal to compare against
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, (list, tuple)) and key.lstrip("-").isdigit() and -len(node) <= int(key) < len(node):
            node = node[int(key)]
        else:
            return _MISSING
    return node
