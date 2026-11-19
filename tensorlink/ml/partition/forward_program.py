"""
tensorlink/ml/partition/forward_program.py

Turns a model's ``forward()`` source into a *linear, cuttable program*.

Why: today only the layer loop is rewritten (injector.py), so every offloaded
chunk is a host -> worker -> host round trip. If the whole forward is a flat list
of statements we can cut it anywhere:

    [host pre] -> worker1 -> worker2 -> ... -> workerK -> [host post]

and each boundary only needs to ship the variables that are *live* across it.

Pipeline
--------
1. ``build_forward_program(model)``
     - parse ``type(model).forward``
     - inline delegations such as ``self.model(...)`` (only into children that,
       directly or transitively, contain a layer loop; leaf modules stay as calls)
     - callee locals are renamed (``_c1_x``) so inlined bodies cannot clobber
       the caller's variables; ``self`` is rebased (``self.x`` -> ``self.model.x``)
2. ``program.bind(model)``   resolve ``self.<path>`` references to real modules
3. ``program.execution_units()``  ordered weight-bearing units (embed, layers, ...)
4. ``program.segment(start, end)``  generate the source of any contiguous slice,
   including slices *inside* the layer loop, with its live-in / live-out names.

A position is ``(stmt_index, layer_k)``. ``layer_k`` only matters when the
statement is the layer loop; otherwise it is 0. ``(n_stmts, 0)`` is the end.

Only the standard library is imported (torch is never needed for the analysis),
so the whole module can be unit-tested without a GPU stack.

Anything this module cannot prove safe raises ``ProgramError``; callers treat that
as "use the fallback strategy".
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

LAYER_ATTRS = frozenset(
    {"layers", "h", "blocks", "decoder_layers", "encoder_layers", "transformer_blocks"}
)
_CONTAINER_TYPES = frozenset({"ModuleList", "Sequential", "ModuleDict"})

Position = Tuple[int, int]


class ProgramError(Exception):
    """forward() cannot be converted to a cuttable linear program."""


# --------------------------------------------------------------------------- #
# small AST helpers
# --------------------------------------------------------------------------- #


def _function_ast(fn) -> ast.FunctionDef:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        raise ProgramError(f"no source available for {fn!r}: {e}") from e
    func = next((n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)), None)
    if func is None:
        raise ProgramError(f"{fn!r} is not a plain function")
    func.decorator_list = []
    return func


def _strip_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _chain_parts(node) -> Optional[tuple]:
    """('model','layers',3) for self.model.layers[3]; None if not rooted at self.
    Only attribute access and constant-int subscripts are followed."""
    parts = []
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        elif (
            isinstance(cur, ast.Subscript)
            and isinstance(cur.slice, ast.Constant)
            and isinstance(cur.slice.value, int)
            and not isinstance(cur.slice.value, bool)
        ):
            parts.append(cur.slice.value)
            cur = cur.value
        else:
            break
    if isinstance(cur, ast.Name) and cur.id == "self" and parts:
        return tuple(reversed(parts))
    return None


def _loads_stores(nodes) -> Tuple[Set[str], Set[str]]:
    loads: Set[str] = set()
    stores: Set[str] = set()
    for root in nodes:
        for n in ast.walk(root):
            if isinstance(n, ast.Name):
                (loads if isinstance(n.ctx, ast.Load) else stores).add(n.id)
            elif isinstance(n, ast.arg):
                stores.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                stores.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    stores.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                stores.add(n.name)
    loads.discard("self")
    stores.discard("self")
    return loads, stores


def _target_names(t) -> Set[str]:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        out: Set[str] = set()
        for e in t.elts:
            out |= _target_names(e)
        return out
    if isinstance(t, ast.Starred):
        return _target_names(t.value)
    return set()


def _exposed_reads(node: ast.For) -> Set[str]:
    """Names a loop reads from *outside* itself (or from a previous iteration).
    The loop target and names assigned before use in the body are loop-local, so
    they must never be treated as live across a cut (they can be module objects)."""
    written = _target_names(node.target)
    exposed = set(_loads_stores([node.iter])[0])
    for st in node.body:
        loads, _ = _loads_stores([st])
        exposed |= loads - written
        if isinstance(st, ast.Assign):
            for t in st.targets:
                written |= _target_names(t)
    return exposed


def _iter_container(it) -> Optional[ast.Attribute]:
    """The ``self.<...>.layers`` node of ``for x in <it>`` or None.
    Accepted forms: C, C[...], enumerate(C), enumerate(C[...])."""
    node = it
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enumerate"
        and node.args
    ):
        node = node.args[0]
    if isinstance(node, ast.Subscript):
        node = node.value
    if (
        isinstance(node, ast.Attribute)
        and node.attr in LAYER_ATTRS
        and _chain_parts(node) is not None
    ):
        return node
    return None


def _slice_iter(it, lo: int, hi: int):
    """Restrict a layer-loop iterable to layers [lo, hi). The slice is applied
    *outside* any original slice, and enumerate() keeps global indices."""
    it = copy.deepcopy(it)

    def mk(c):
        return ast.Subscript(
            value=c,
            slice=ast.Slice(lower=ast.Constant(lo), upper=ast.Constant(hi)),
            ctx=ast.Load(),
        )

    if isinstance(it, ast.Call):  # enumerate(...)
        it.args[0] = mk(it.args[0])
        if len(it.args) == 1 and not any(k.arg == "start" for k in it.keywords):
            it.keywords.append(ast.keyword(arg="start", value=ast.Constant(lo)))
        return it
    return mk(it)


class _SelfRebase(ast.NodeTransformer):
    """self -> self.<prefix...>"""

    def __init__(self, prefix: List[str]):
        self.prefix = prefix

    def visit_Name(self, node):
        if node.id == "self" and isinstance(node.ctx, ast.Load):
            new = ast.Name(id="self", ctx=ast.Load())
            for p in self.prefix:
                new = ast.Attribute(value=new, attr=p, ctx=ast.Load())
            return new
        return node


class _Rename(ast.NodeTransformer):
    def __init__(self, mapping: Dict[str, str]):
        self.m = mapping

    def visit_Name(self, n):
        n.id = self.m.get(n.id, n.id)
        return n

    def visit_arg(self, n):
        n.arg = self.m.get(n.arg, n.arg)
        return n

    def visit_ExceptHandler(self, n):
        if n.name:
            n.name = self.m.get(n.name, n.name)
        self.generic_visit(n)
        return n


class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, target, replacement):
        self.target, self.replacement = target, replacement

    def visit(self, node):
        if node is self.target:
            return self.replacement
        return self.generic_visit(node)


class _Normalize(ast.NodeTransformer):
    """`x: T = v` -> `x = v`; bare `x: T` -> pass."""

    def visit_AnnAssign(self, n):
        self.generic_visit(n)
        if n.value is None:
            return ast.Pass()
        return ast.Assign(targets=[n.target], value=n.value, lineno=getattr(n, "lineno", 0))


class _Refs(ast.NodeVisitor):
    def __init__(self):
        self.refs: List[tuple] = []

    def _visit_chain(self, node):
        chain = _chain_parts(node)
        if chain is not None:
            self.refs.append(chain)
        else:
            self.generic_visit(node)

    visit_Attribute = _visit_chain
    visit_Subscript = _visit_chain


# --------------------------------------------------------------------------- #
# inlining
# --------------------------------------------------------------------------- #


class _Ctx:
    def __init__(self, is_module: Callable, max_depth: int):
        self.is_module = is_module
        self.max_depth = max_depth
        self.counter = 0
        self.classes: List[type] = []
        self._loop_cache: Dict[type, bool] = {}


def _simple(st) -> bool:
    return isinstance(st, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return, ast.AugAssign))


def _module_calls(st, owner, ctx: _Ctx):
    for n in ast.walk(st):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"
        ):
            child = getattr(owner, n.func.attr, None)
            if child is not None and ctx.is_module(child):
                yield n, n.func.attr, child


def _has_layer_loop(child, ctx: _Ctx, depth: int) -> bool:
    cls = type(child)
    if cls in ctx._loop_cache:
        return ctx._loop_cache[cls]
    ctx._loop_cache[cls] = False  # recursion guard
    try:
        func = _function_ast(cls.forward)
    except ProgramError:
        return False
    found = any(isinstance(n, ast.For) and _iter_container(n.iter) for n in ast.walk(func))
    if not found and depth < ctx.max_depth:
        for st in func.body:
            if _simple(st):
                for _, _, sub in _module_calls(st, child, ctx):
                    if _has_layer_loop(sub, ctx, depth + 1):
                        found = True
    ctx._loop_cache[cls] = found
    return found


def _flatten_body(body, owner, ctx: _Ctx, depth: int) -> List[ast.stmt]:
    out: List[ast.stmt] = []
    for st in body:
        hit = None
        if depth < ctx.max_depth and _simple(st):
            for call, attr, child in _module_calls(st, owner, ctx):
                if _has_layer_loop(child, ctx, depth + 1):
                    hit = (call, attr, child)
                    break
        if hit is None:
            out.append(st)
        else:
            out.extend(_inline_call(st, *hit, ctx, depth))
    return out


def _inline_call(st, call, attr, child, ctx: _Ctx, depth: int) -> List[ast.stmt]:
    ctx.counter += 1
    tag = f"_c{ctx.counter}_"
    cfunc = _function_ast(type(child).forward)
    ctx.classes.append(type(child))
    a = cfunc.args
    if a.vararg or a.posonlyargs:
        raise ProgramError(f"{type(child).__name__}.forward uses *args/pos-only params")

    pos_params = [x.arg for x in a.args][1:]  # drop self
    kwonly = [x.arg for x in a.kwonlyargs]
    params = pos_params + kwonly
    defaults: Dict[str, ast.expr] = {}
    nd = len(a.defaults)
    for i, p in enumerate(pos_params):
        di = i - (len(pos_params) - nd)
        if di >= 0:
            defaults[p] = a.defaults[di]
    for p, d in zip(kwonly, a.kw_defaults):
        if d is not None:
            defaults[p] = d

    if any(isinstance(x, ast.Starred) for x in call.args):
        raise ProgramError("starred positional args in delegated call")
    bound: Dict[str, ast.expr] = {}
    for p, v in zip(params, call.args):
        bound[p] = v
    star: List[ast.expr] = []
    extra: List[ast.keyword] = []
    for kw in call.keywords:
        if kw.arg is None:
            star.append(kw.value)
        elif kw.arg in params:
            bound[kw.arg] = kw.value
        else:
            extra.append(kw)
    if len(star) > 1:
        raise ProgramError("multiple **kwargs expansions in delegated call")
    star_src = ast.unparse(star[0]) if star else None

    # callee body: strip docstring, inline *its* delegates first
    cbody = _flatten_body(_strip_docstring(cfunc.body), child, ctx, depth + 1)

    # returns: only the final top-level statement may return
    rets = [n for s in cbody for n in ast.walk(s) if isinstance(n, ast.Return)]
    last_is_ret = bool(cbody) and isinstance(cbody[-1], ast.Return)
    if len(rets) > (1 if last_is_ret else 0):
        raise ProgramError(f"{type(child).__name__}.forward has early/nested returns")

    # rename callee locals
    loads, stores = _loads_stores(cbody)
    local = (set(params) | stores | ({a.kwarg.arg} if a.kwarg else set())) - {"self"}
    mapping = {n: tag + n for n in local}
    cbody = [_Rename(mapping).visit(s) for s in cbody]
    cbody = [_SelfRebase([attr]).visit(s) for s in cbody]

    retvar = tag + "ret"
    if last_is_ret:
        r = cbody[-1]
        cbody[-1] = ast.Assign(
            targets=[ast.Name(id=retvar, ctx=ast.Store())],
            value=r.value if r.value is not None else ast.Constant(None),
            lineno=0,
        )
    else:
        cbody.append(
            ast.Assign(
                targets=[ast.Name(id=retvar, ctx=ast.Store())],
                value=ast.Constant(None),
                lineno=0,
            )
        )

    # parameter binding (evaluated in the caller's namespace)
    pro: List[str] = []
    for p in params:
        if p in bound:
            val = ast.unparse(bound[p])
        elif star_src is not None:
            if p in defaults:
                val = f"{star_src}.get({p!r}, {ast.unparse(defaults[p])})"
            else:
                val = f"{star_src}[{p!r}]"
        elif p in defaults:
            val = ast.unparse(defaults[p])
        else:
            raise ProgramError(f"missing argument {p!r} for {type(child).__name__}.forward")
        pro.append(f"{tag}{p} = {val}")
    if a.kwarg:
        entries = ", ".join(f"{k.arg!r}: {ast.unparse(k.value)}" for k in extra)
        if star_src is not None:
            pro.append(
                f"{tag}{a.kwarg.arg} = {{**{{k: v for k, v in {star_src}.items() "
                f"if k not in {tuple(params)!r}}}, {entries}}}"
            )
        else:
            pro.append(f"{tag}{a.kwarg.arg} = {{{entries}}}")
    elif extra:
        raise ProgramError("unexpected keyword args for delegated call")
    binding = [ast.parse(s).body[0] for s in pro]

    new_st = _ReplaceNode(call, ast.Name(id=retvar, ctx=ast.Load())).visit(st)
    return binding + cbody + [new_st]


# --------------------------------------------------------------------------- #
# program model
# --------------------------------------------------------------------------- #


@dataclass
class LoopInfo:
    container: str = ""  # e.g. "model.layers"
    n: int = 0
    shared: List[str] = field(default_factory=list)  # modules used inside the body


@dataclass
class Stmt:
    node: ast.stmt
    reads: Set[str]
    writes: Set[str]
    kills: Set[str]
    refs: List[tuple]
    loop: Optional[LoopInfo] = None
    module_paths: List[str] = field(default_factory=list)


@dataclass
class ExecUnit:
    """One weight-bearing step of the forward pass, in execution order."""

    paths: Tuple[str, ...]  # module paths (root-relative, e.g. "model.layers.3")
    pos: Position  # where this unit's statement starts
    end: Position  # exclusive end of the unit's own statement (normalized)
    stmt: int
    layer: Optional[int] = None

    @property
    def key(self) -> str:
        return self.paths[0]


@dataclass
class SegmentCode:
    source: str
    live_in: List[str]
    live_out: List[str]
    final: bool  # contains the program's return statement
    modules: List[str]  # module paths this segment must have loaded
    has_layers: bool  # runs part of the layer loop (advances KV cache)
    sha: str


def _default_is_module(o) -> bool:
    try:
        import torch.nn as nn

        return isinstance(o, nn.Module)
    except Exception:  # pragma: no cover
        return hasattr(o, "forward")


def _walk_path(root, parts):
    obj = root
    for p in parts:
        obj = obj[p] if isinstance(p, int) else getattr(obj, p)
    return obj


class ForwardProgram:
    def __init__(self, params, sig_src, kwarg, stmts, classes, local_names, flat_source):
        self.params: List[str] = params
        self.sig_src: str = sig_src  # without leading "self, "
        self.kwarg: Optional[str] = kwarg
        self.stmts: List[Stmt] = stmts
        self.classes: List[type] = classes
        self.local_names: Set[str] = local_names
        self.flat_source: str = flat_source
        self.fingerprint = hashlib.sha256(flat_source.encode()).hexdigest()
        self._bound = False
        self._compute_liveness()

    # ----- liveness ---------------------------------------------------------
    def _compute_liveness(self):
        n = len(self.stmts)
        self.live_before: List[Set[str]] = [set() for _ in range(n + 1)]
        self.live_after: List[Set[str]] = [set() for _ in range(n)]
        live: Set[str] = set()
        for i in range(n - 1, -1, -1):
            self.live_after[i] = set(live)
            st = self.stmts[i]
            live = (live - st.kills) | st.reads
            self.live_before[i] = set(live)
        self.defined_before: List[Set[str]] = []
        d = set(self.params)
        for st in self.stmts:
            self.defined_before.append(set(d))
            d |= st.writes
        self.defined_before.append(set(d))

    def boundary(self, pos: Position) -> List[str]:
        """Variables that must cross a cut placed at ``pos``."""
        i, k = pos
        n = len(self.stmts)
        if i >= n:
            return []
        st = self.stmts[i]
        if st.loop and k > 0:
            need = self.live_before[i] | self.live_after[i]
            have = self.defined_before[i] | st.writes
        else:
            need = self.live_before[i]
            have = self.defined_before[i]
        return sorted(need & have)

    # ----- binding to a concrete model ---------------------------------------
    def bind(self, model, is_module: Optional[Callable] = None):
        is_module = is_module or _default_is_module
        seen: Dict[str, int] = {}
        for i, st in enumerate(self.stmts):
            if st.loop is not None:
                cont = _walk_path(model, self._container_parts(st))
                if type(cont).__name__ not in _CONTAINER_TYPES and not isinstance(cont, (list, tuple)):
                    raise ProgramError(f"layer container {st.loop.container!r} is not a module list")
                st.loop.n = self._loop_len(st, model, cont)
                st.loop.shared = self._resolve_all(model, st.refs, is_module, allow_container=False)
                st.module_paths = []
            else:
                st.module_paths = self._resolve_all(model, st.refs, is_module, allow_container=False)
                # A module may be called from several statements (e.g. GPT-2's wte is
                # applied to input_ids and to token_type_ids). The first statement owns
                # it for planning; every segment that runs a statement loads the modules
                # that statement references (see _segment_nodes), so a repeat is safe.
                for p in st.module_paths:
                    seen.setdefault(p, i)
        self._bound = True
        return self

    @staticmethod
    def _container_parts(st: Stmt):
        return tuple(int(p) if p.isdigit() else p for p in st.loop.container.split("."))

    @staticmethod
    def _loop_len(st: Stmt, model, container) -> int:
        try:
            it = copy.deepcopy(st.node.iter)
            expr = ast.Expression(ast.fix_missing_locations(it))
            val = eval(
                compile(expr, "<iter>", "eval"),
                {"self": model, "enumerate": enumerate, "list": list},
            )
            return len(list(val))
        except Exception:
            return len(container)

    @staticmethod
    def _resolve_all(model, refs, is_module, allow_container) -> List[str]:
        out: List[str] = []
        for chain in refs:
            path = ForwardProgram._resolve_ref(model, chain, is_module)
            if path and path not in out:
                out.append(path)
        # keep only outermost when a module and its descendant are both referenced
        return [p for p in out if not any(q != p and p.startswith(q + ".") for q in out)]

    @staticmethod
    def _resolve_ref(model, chain, is_module) -> Optional[str]:
        obj, path = model, []
        last_mod = None
        for part in chain:
            prev = obj
            try:
                obj = obj[part] if isinstance(part, int) else getattr(obj, part)
            except Exception:
                return None
            if is_module(obj):
                path.append(str(part))
                last_mod = ".".join(path)
                if type(obj).__name__ in _CONTAINER_TYPES:
                    raise ProgramError(
                        f"statement indexes module container {last_mod!r} outside a layer loop"
                    )
            else:
                owned = getattr(prev, "_parameters", {}) or {}
                bufs = getattr(prev, "_buffers", {}) or {}
                if isinstance(part, str) and (part in owned or part in bufs):
                    return last_mod  # e.g. self.lm_head.weight -> lm_head
                return None  # config, flags, methods
        return last_mod

    # ----- execution units ---------------------------------------------------
    def execution_units(self) -> List[ExecUnit]:
        assert self._bound, "call bind(model) first"
        units: List[ExecUnit] = []
        n = len(self.stmts)
        owned: Set[str] = set()
        for i, st in enumerate(self.stmts):
            if st.loop is not None:
                for k in range(st.loop.n):
                    end = (i, k + 1) if k + 1 < st.loop.n else (i + 1, 0)
                    units.append(
                        ExecUnit(
                            paths=(f"{st.loop.container}.{k}",),
                            pos=(i, k),
                            end=end,
                            stmt=i,
                            layer=k,
                        )
                    )
            elif st.module_paths:
                new = tuple(p for p in st.module_paths if p not in owned)
                owned.update(st.module_paths)
                if new:  # pure repeats add no planning unit; they ride with the next one
                    units.append(ExecUnit(paths=new, pos=(i, 0), end=(i + 1, 0), stmt=i))
        return units

    @property
    def end_pos(self) -> Position:
        return (len(self.stmts), 0)

    # ----- segments ----------------------------------------------------------
    def _segment_nodes(self, start: Position, end: Position):
        i0, k0 = start
        i1, k1 = end
        n = len(self.stmts)
        if start > end:
            raise ProgramError(f"bad segment {start}..{end}")
        out, has_layers, modules = [], False, []
        for idx in range(i0, min(i1, n - 1) + 1):
            st = self.stmts[idx]
            if idx == i1 and not (st.loop and k1 > 0):
                break
            if st.loop is not None:
                lo = k0 if idx == i0 else 0
                hi = k1 if idx == i1 else st.loop.n
                if hi <= lo:
                    continue
                node = copy.deepcopy(st.node)
                if not (lo == 0 and hi == st.loop.n):
                    if _iter_container(node.iter) is None:  # pragma: no cover
                        raise ProgramError("cannot split this loop form")
                    node.iter = _slice_iter(node.iter, lo, hi)
                out.append(node)
                has_layers = True
                modules += [f"{st.loop.container}.{k}" for k in range(lo, hi)]
                modules += [m for m in st.loop.shared if m not in modules]
            else:
                if idx == i0 and k0:
                    raise ProgramError("cannot start mid-statement outside the layer loop")
                out.append(copy.deepcopy(st.node))
                modules += [m for m in st.module_paths if m not in modules]
        return out, has_layers, modules

    def segment(self, start: Position, end: Position) -> SegmentCode:
        assert self._bound, "call bind(model) first"
        nodes, has_layers, modules = self._segment_nodes(start, end)
        n = len(self.stmts)
        final = end[0] >= n
        live_in = self.boundary(start) if start[0] < n else []
        live_out = [] if final else self.boundary(end)

        class _Ret(ast.NodeTransformer):
            def visit_Return(self, r):
                val = r.value if r.value is not None else ast.Constant(None)
                return ast.Return(
                    value=ast.Dict(keys=[ast.Constant("__return__")], values=[val])
                )

        if final:
            nodes = [_Ret().visit(x) for x in nodes]
        lines = ["def segment_forward(self, _in):"]
        for v in live_in:
            lines.append(f"    {v} = _in.get({v!r})")
        body = "\n".join(ast.unparse(ast.fix_missing_locations(x)) for x in nodes)
        lines.append(textwrap.indent(body, "    ") if body else "    pass")
        if not final:
            lines.append("    _l = locals()")
            lines.append(
                f"    return {{'__vars__': {{_k: _l[_k] for _k in {live_out!r} if _k in _l}}}}"
            )
        src = "\n".join(lines) + "\n"
        return SegmentCode(
            source=src,
            live_in=live_in,
            live_out=live_out,
            final=final,
            modules=modules,
            has_layers=has_layers,
            sha=hashlib.sha256(src.encode()).hexdigest(),
        )

    def namespace(self) -> dict:
        """Globals for exec()'d code: the defining modules of every inlined class."""
        ns: dict = {}
        for cls in reversed(self.classes):
            mod = sys.modules.get(cls.__module__)
            if mod is not None:
                ns.update(vars(mod))
        ns.pop("__name__", None)
        return ns

    def compile_segment(self, code: SegmentCode) -> Callable:
        ns = self.namespace()
        exec(compile(code.source, "<tensorlink-segment>", "exec"), ns)
        return ns["segment_forward"]

    def host_forward_source(self, runner_attr: str = "_tl_pipeline") -> str:
        """Replacement ``forward`` that keeps the original signature (HF generate()
        inspects it) and hands the bound arguments to the pipeline runner."""
        items = ", ".join(f"{p!r}: {p}" for p in self.params)
        return (
            f"def forward(self, {self.sig_src}):\n"
            f"    return self.{runner_attr}.run({{{items}}})\n"
        )


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #


def _signature_src(fn: ast.FunctionDef) -> Tuple[str, List[str], Optional[str]]:
    a = fn.args
    if a.vararg:
        raise ProgramError("root forward() takes *args")
    pos = a.posonlyargs + a.args
    nd = len(a.defaults)
    parts, names = [], []
    for i, arg in enumerate(pos):
        s = arg.arg
        di = i - (len(pos) - nd)
        if di >= 0:
            s += "=" + ast.unparse(a.defaults[di])
        parts.append(s)
        names.append(arg.arg)
    if a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + (f"={ast.unparse(d)}" if d is not None else ""))
        names.append(arg.arg)
    kwarg = a.kwarg.arg if a.kwarg else None
    if kwarg:
        parts.append(f"**{kwarg}")
        names.append(kwarg)
    if not names or names[0] != "self":
        raise ProgramError("forward() has no self")
    return ", ".join(parts[1:]), names[1:], kwarg


def build_forward_program(
    model, *, is_module: Optional[Callable] = None, max_inline_depth: int = 3
) -> ForwardProgram:
    is_module = is_module or _default_is_module
    ctx = _Ctx(is_module, max_inline_depth)
    root_fn = _function_ast(type(model).forward)
    ctx.classes.append(type(model))
    sig_src, params, kwarg = _signature_src(root_fn)

    body = _flatten_body(_strip_docstring(root_fn.body), model, ctx, 0)
    body = [_Normalize().visit(s) for s in body]
    body = [s for s in body if not isinstance(s, ast.Pass)] or body

    if not body or not isinstance(body[-1], ast.Return):
        raise ProgramError("forward() must end with a top-level return")
    for s in body[:-1]:
        for n in ast.walk(s):
            if isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal)):
                raise ProgramError(f"unsupported construct {type(n).__name__} in forward()")

    local_names: Set[str] = set(params)
    for s in body:
        local_names |= _loads_stores([s])[1]

    stmts: List[Stmt] = []
    for node in body:
        loads, stores = _loads_stores([node])
        kills: Set[str] = set()
        if isinstance(node, ast.Assign):
            for t in node.targets:
                kills |= _target_names(t)
        loop = None
        refs_root = [node]
        if isinstance(node, ast.For):
            cont = _iter_container(node.iter)
            if cont is not None:
                loop = LoopInfo(container=".".join(str(p) for p in _chain_parts(cont)))
                refs_root = list(node.body) + list(node.orelse)
                loads = _exposed_reads(node)
        else:
            for sub in ast.walk(node):
                if isinstance(sub, ast.For) and _iter_container(sub.iter) is not None:
                    raise ProgramError("layer loop is nested inside another statement")
        v = _Refs()
        for r in refs_root:
            v.visit(r)
        stmts.append(
            Stmt(
                node=node,
                reads=loads & local_names,
                writes=stores,
                kills=kills,
                refs=v.refs,
                loop=loop,
            )
        )

    flat_src = f"def forward(self, {sig_src}):\n" + textwrap.indent(
        "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in body), "    "
    )
    return ForwardProgram(
        params=params,
        sig_src=sig_src,
        kwarg=kwarg,
        stmts=stmts,
        classes=ctx.classes,
        local_names=local_names,
        flat_source=flat_src,
    )
