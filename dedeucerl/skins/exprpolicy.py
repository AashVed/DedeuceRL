"""ExprPolicyEnv: Goal-conditioned typed policy DSL debugging skin.

This skin is intentionally *not* a pure "exploration" / system-identification task.
It models a common real-world agent workflow:

- You are given a typed expression DSL (policy/config language) plus a request schema.
- You are given a buggy starter policy expression.
- You iterate using compiler feedback (type_check) and test feedback (run_tests).
- You must submit a corrected expression that generalizes to hidden tests.

Why this matters
----------------
Modern code agents often integrate with internal DSLs (feature flags, ABAC rules,
workflow configs). In practice, learning happens via:

- rich diagnostics (expected/found types)
- failing examples (unit/CI tests)

This environment preserves those realities, and keeps the task non-trivial via:
- hidden tests (unseen cases)
- realistic tool costs (CI is expensive)

The DSL itself is small and safe: expressions are parsed and evaluated by a
custom interpreter (no Python eval).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from dedeucerl.core.config import SkinConfig
from dedeucerl.core.domain_spec import (
    ArgSchema,
    DomainSpec,
    ObservationField,
    ParamSpec,
    ReturnField,
    ToolSchema,
)
from dedeucerl.core.env import HiddenSystemEnv
from dedeucerl.utils import error_invalid_argument
from dedeucerl.utils.rng import get_rng


# -----------------------------------------------------------------------------
# Skin configuration
# -----------------------------------------------------------------------------


EXPRPOLICY_CONFIG = SkinConfig(
    isomorphism_check=False,
    trap_enabled=True,
    trap_ends_episode=True,
    default_budget=60,
    submission_cost=12,
    max_turns=120,
    skin_name="exprpolicy",
    skin_version="1.0",
)


# -----------------------------------------------------------------------------
# DSL: Types, AST, Parser
# -----------------------------------------------------------------------------


TypeName = str  # "Bool" | "Int" | "String"

T_BOOL: TypeName = "Bool"
T_INT: TypeName = "Int"
T_STR: TypeName = "String"


@dataclass(frozen=True)
class CompileError:
    error_kind: str  # ParseError, TypeMismatch, UnknownName, ...
    message: str
    expected: Optional[str] = None
    found: Optional[str] = None
    at: Optional[str] = None
    pos: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "error_kind": self.error_kind,
            "message": self.message,
        }
        if self.expected is not None:
            out["expected"] = self.expected
        if self.found is not None:
            out["found"] = self.found
        if self.at is not None:
            out["at"] = self.at
        if self.pos is not None:
            out["pos"] = int(self.pos)
        return out


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


class _Lexer:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def _peek(self) -> str:
        if self.i >= len(self.s):
            return ""
        return self.s[self.i]

    def _consume_ws(self) -> None:
        while self._peek() and self._peek().isspace():
            self.i += 1

    def next_token(self) -> Token:
        self._consume_ws()
        pos = self.i
        ch = self._peek()
        if not ch:
            return Token("EOF", "", pos)

        # Two-char operators
        if self.s.startswith("&&", self.i):
            self.i += 2
            return Token("OP", "&&", pos)
        if self.s.startswith("||", self.i):
            self.i += 2
            return Token("OP", "||", pos)
        if self.s.startswith("==", self.i):
            self.i += 2
            return Token("OP", "==", pos)
        if self.s.startswith("!=", self.i):
            self.i += 2
            return Token("OP", "!=", pos)

        # Single-char tokens
        if ch in "!(),":
            self.i += 1
            return Token(ch, ch, pos)

        # String literal (simple, no escapes)
        if ch == '"':
            self.i += 1
            start = self.i
            while self._peek() and self._peek() != '"':
                self.i += 1
            if not self._peek():
                return Token("BAD", self.s[start:], pos)
            text = self.s[start : self.i]
            self.i += 1  # closing quote
            return Token("STRING", text, pos)

        # Integer literal
        if ch.isdigit():
            start = self.i
            while self._peek() and self._peek().isdigit():
                self.i += 1
            return Token("INT", self.s[start : self.i], pos)

        # Identifier / keywords
        m = _IDENT_RE.match(self.s, self.i)
        if m:
            text = m.group(0)
            self.i = m.end()
            if text in ("true", "false"):
                return Token("BOOL", text, pos)
            return Token("IDENT", text, pos)

        # Unknown character
        self.i += 1
        return Token("BAD", ch, pos)


class ParseError(Exception):
    def __init__(self, message: str, pos: int):
        super().__init__(message)
        self.message = message
        self.pos = pos


class Expr:
    pass


@dataclass(frozen=True)
class BoolLit(Expr):
    value: bool


@dataclass(frozen=True)
class IntLit(Expr):
    value: int


@dataclass(frozen=True)
class StrLit(Expr):
    value: str


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str  # '!'
    expr: Expr


@dataclass(frozen=True)
class BinaryOp(Expr):
    op: str  # '&&' | '||' | '==' | '!='
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Call(Expr):
    name: str
    args: Tuple[Expr, ...]


class _Parser:
    def __init__(self, s: str):
        self._lex = _Lexer(s)
        self._cur = self._lex.next_token()
        self._s = s

    def _eat(self, kind: str, text: Optional[str] = None) -> Token:
        tok = self._cur
        if tok.kind != kind:
            raise ParseError(f"Expected {kind}, got {tok.kind}", tok.pos)
        if text is not None and tok.text != text:
            raise ParseError(f"Expected '{text}', got '{tok.text}'", tok.pos)
        self._cur = self._lex.next_token()
        return tok

    def _match(self, kind: str, text: Optional[str] = None) -> bool:
        if self._cur.kind != kind:
            return False
        if text is not None and self._cur.text != text:
            return False
        return True

    def parse(self) -> Expr:
        expr = self._parse_or()
        if self._cur.kind != "EOF":
            raise ParseError(f"Unexpected token '{self._cur.text}'", self._cur.pos)
        return expr

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._match("OP", "||"):
            self._eat("OP", "||")
            right = self._parse_and()
            left = BinaryOp("||", left, right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        while self._match("OP", "&&"):
            self._eat("OP", "&&")
            right = self._parse_not()
            left = BinaryOp("&&", left, right)
        return left

    def _parse_not(self) -> Expr:
        if self._match("!", "!"):
            self._eat("!", "!")
            return UnaryOp("!", self._parse_not())
        return self._parse_cmp()

    def _parse_cmp(self) -> Expr:
        left = self._parse_primary()
        if self._match("OP", "=="):
            self._eat("OP", "==")
            right = self._parse_primary()
            return BinaryOp("==", left, right)
        if self._match("OP", "!="):
            self._eat("OP", "!=")
            right = self._parse_primary()
            return BinaryOp("!=", left, right)
        return left

    def _parse_primary(self) -> Expr:
        tok = self._cur

        if tok.kind == "BOOL":
            self._eat("BOOL")
            return BoolLit(tok.text == "true")
        if tok.kind == "INT":
            self._eat("INT")
            return IntLit(int(tok.text))
        if tok.kind == "STRING":
            self._eat("STRING")
            return StrLit(tok.text)

        if tok.kind == "IDENT":
            self._eat("IDENT")
            # Function call
            if self._match("(", "("):
                self._eat("(", "(")
                args: List[Expr] = []
                if not self._match(")", ")"):
                    args.append(self._parse_or())
                    while self._match(",", ","):
                        self._eat(",", ",")
                        args.append(self._parse_or())
                self._eat(")", ")")
                return Call(tok.text, tuple(args))
            return Var(tok.text)

        if tok.kind == "(":
            self._eat("(", "(")
            expr = self._parse_or()
            self._eat(")", ")")
            return expr

        raise ParseError(f"Unexpected token '{tok.text}'", tok.pos)


# -----------------------------------------------------------------------------
# DSL: Typechecking + Evaluation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltinSig:
    name: str
    args: Tuple[TypeName, ...]
    ret: TypeName
    description: str = ""

    def to_prompt_sig(self) -> str:
        args = ", ".join(self.args)
        return f"{self.name}({args}) -> {self.ret}"


def _ast_size(expr: Expr) -> int:
    if isinstance(expr, (BoolLit, IntLit, StrLit, Var)):
        return 1
    if isinstance(expr, UnaryOp):
        return 1 + _ast_size(expr.expr)
    if isinstance(expr, BinaryOp):
        return 1 + _ast_size(expr.left) + _ast_size(expr.right)
    if isinstance(expr, Call):
        return 1 + sum(_ast_size(a) for a in expr.args)
    return 1


def _typecheck(
    expr: Expr,
    *,
    field_schema: Dict[str, TypeName],
    builtins: Dict[str, BuiltinSig],
) -> Tuple[Optional[TypeName], List[CompileError]]:
    errors: List[CompileError] = []

    def rec(e: Expr) -> Optional[TypeName]:
        if isinstance(e, BoolLit):
            return T_BOOL
        if isinstance(e, IntLit):
            return T_INT
        if isinstance(e, StrLit):
            return T_STR
        if isinstance(e, Var):
            t = field_schema.get(e.name)
            if t is None:
                errors.append(
                    CompileError(
                        error_kind="UnknownName",
                        message=f"Unknown identifier '{e.name}'",
                        at=e.name,
                    )
                )
                return None
            return t
        if isinstance(e, UnaryOp):
            t = rec(e.expr)
            if t is None:
                return None
            if e.op == "!":
                if t != T_BOOL:
                    errors.append(
                        CompileError(
                            error_kind="TypeMismatch",
                            message="Operator '!' expects Bool",
                            expected=T_BOOL,
                            found=t,
                            at="!",
                        )
                    )
                    return None
                return T_BOOL
            errors.append(
                CompileError(
                    error_kind="UnknownOperator",
                    message=f"Unknown unary operator '{e.op}'",
                    at=e.op,
                )
            )
            return None
        if isinstance(e, BinaryOp):
            tl = rec(e.left)
            tr = rec(e.right)
            if tl is None or tr is None:
                return None
            if e.op in ("&&", "||"):
                if tl != T_BOOL or tr != T_BOOL:
                    errors.append(
                        CompileError(
                            error_kind="TypeMismatch",
                            message=f"Operator '{e.op}' expects Bool && Bool",
                            expected=f"{T_BOOL},{T_BOOL}",
                            found=f"{tl},{tr}",
                            at=e.op,
                        )
                    )
                    return None
                return T_BOOL
            if e.op in ("==", "!="):
                if tl != tr:
                    errors.append(
                        CompileError(
                            error_kind="TypeMismatch",
                            message=f"Operator '{e.op}' expects both sides to have the same type",
                            expected=tl,
                            found=tr,
                            at=e.op,
                        )
                    )
                    return None
                return T_BOOL
            errors.append(
                CompileError(
                    error_kind="UnknownOperator",
                    message=f"Unknown operator '{e.op}'",
                    at=e.op,
                )
            )
            return None
        if isinstance(e, Call):
            sig = builtins.get(e.name)
            if sig is None:
                errors.append(
                    CompileError(
                        error_kind="UnknownFunction",
                        message=f"Unknown function '{e.name}'",
                        at=e.name,
                    )
                )
                return None
            if len(e.args) != len(sig.args):
                errors.append(
                    CompileError(
                        error_kind="ArityError",
                        message=f"Function '{e.name}' expects {len(sig.args)} args, got {len(e.args)}",
                        expected=str(len(sig.args)),
                        found=str(len(e.args)),
                        at=e.name,
                    )
                )
                return None
            for i, (arg_expr, expected_t) in enumerate(zip(e.args, sig.args)):
                ta = rec(arg_expr)
                if ta is None:
                    return None
                if ta != expected_t:
                    errors.append(
                        CompileError(
                            error_kind="TypeMismatch",
                            message=f"Argument {i} of '{e.name}' expects {expected_t}",
                            expected=expected_t,
                            found=ta,
                            at=e.name,
                        )
                    )
                    return None
            return sig.ret
        errors.append(CompileError(error_kind="InternalError", message="Unknown AST node"))
        return None

    t = rec(expr)
    return t, errors


def _eval_expr(
    expr: Expr,
    *,
    env: Dict[str, Any],
    builtins: Dict[str, BuiltinSig],
) -> Any:
    # Type safety is enforced by type_check before evaluation.
    if isinstance(expr, BoolLit):
        return bool(expr.value)
    if isinstance(expr, IntLit):
        return int(expr.value)
    if isinstance(expr, StrLit):
        return str(expr.value)
    if isinstance(expr, Var):
        return env.get(expr.name)
    if isinstance(expr, UnaryOp):
        if expr.op == "!":
            return not bool(_eval_expr(expr.expr, env=env, builtins=builtins))
        raise RuntimeError(f"Unknown unary operator {expr.op}")
    if isinstance(expr, BinaryOp):
        if expr.op == "&&":
            return bool(_eval_expr(expr.left, env=env, builtins=builtins)) and bool(
                _eval_expr(expr.right, env=env, builtins=builtins)
            )
        if expr.op == "||":
            return bool(_eval_expr(expr.left, env=env, builtins=builtins)) or bool(
                _eval_expr(expr.right, env=env, builtins=builtins)
            )
        if expr.op == "==":
            return _eval_expr(expr.left, env=env, builtins=builtins) == _eval_expr(
                expr.right, env=env, builtins=builtins
            )
        if expr.op == "!=":
            return _eval_expr(expr.left, env=env, builtins=builtins) != _eval_expr(
                expr.right, env=env, builtins=builtins
            )
        raise RuntimeError(f"Unknown operator {expr.op}")
    if isinstance(expr, Call):
        fn = expr.name
        args = [_eval_expr(a, env=env, builtins=builtins) for a in expr.args]

        if fn == "contains":
            return str(args[0]).find(str(args[1])) >= 0
        if fn == "starts_with":
            return str(args[0]).startswith(str(args[1]))
        if fn == "lower":
            return str(args[0]).lower()
        if fn == "is_internal_ip":
            ip = str(args[0])
            return ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("172.16.")
        raise RuntimeError(f"Unknown builtin '{fn}'")
    raise RuntimeError("Unknown AST node")


def _expr_to_str(expr: Expr) -> str:
    # Use explicit parentheses to keep formatting stable.
    if isinstance(expr, BoolLit):
        return "true" if expr.value else "false"
    if isinstance(expr, IntLit):
        return str(expr.value)
    if isinstance(expr, StrLit):
        # No escapes in v1.
        return '"' + expr.value.replace('"', "") + '"'
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, UnaryOp):
        return f"(!{_expr_to_str(expr.expr)})"
    if isinstance(expr, BinaryOp):
        return f"({_expr_to_str(expr.left)} {expr.op} {_expr_to_str(expr.right)})"
    if isinstance(expr, Call):
        args = ", ".join(_expr_to_str(a) for a in expr.args)
        return f"{expr.name}({args})"
    return "<expr>"


def _compile(
    expr_str: str,
    *,
    field_schema: Dict[str, TypeName],
    builtins: Dict[str, BuiltinSig],
) -> Tuple[Optional[Expr], Optional[TypeName], List[CompileError]]:
    try:
        ast = _Parser(expr_str).parse()
    except ParseError as e:
        return None, None, [CompileError("ParseError", e.message, pos=e.pos)]
    except Exception:
        return None, None, [CompileError("ParseError", "Failed to parse expression")]

    t, errors = _typecheck(ast, field_schema=field_schema, builtins=builtins)
    if errors:
        return ast, None, errors
    return ast, t, []


# -----------------------------------------------------------------------------
# Task generation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    input: Dict[str, Any]
    expected: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"input": self.input, "expected": bool(self.expected)}


def _default_field_schema() -> Dict[str, TypeName]:
    # Flat keys, but dotted identifiers are allowed to feel like real request structs.
    return {
        "req.user.role": T_STR,
        "req.ip": T_STR,
        "req.resource": T_STR,
        "req.method": T_STR,
    }


def _default_builtins() -> Dict[str, BuiltinSig]:
    return {
        "contains": BuiltinSig(
            name="contains",
            args=(T_STR, T_STR),
            ret=T_BOOL,
            description="contains(haystack, needle) -> Bool",
        ),
        "starts_with": BuiltinSig(
            name="starts_with",
            args=(T_STR, T_STR),
            ret=T_BOOL,
            description="starts_with(s, prefix) -> Bool",
        ),
        "lower": BuiltinSig(
            name="lower",
            args=(T_STR,),
            ret=T_STR,
            description="lower(s) -> String",
        ),
        "is_internal_ip": BuiltinSig(
            name="is_internal_ip",
            args=(T_STR,),
            ret=T_BOOL,
            description="is_internal_ip(ip) -> Bool",
        ),
    }


def _sample_request(rng: Any) -> Dict[str, Any]:
    roles = ["admin", "employee", "contractor", "guest"]
    methods = ["GET", "POST", "DELETE"]
    resources = [
        "finance",
        "hr",
        "engineering",
        "public/docs",
        "public/blog",
        "internal/tools",
    ]
    # Small, realistic IP pool.
    internal_ips = ["10.0.0.5", "192.168.1.12", "172.16.3.9"]
    external_ips = ["8.8.8.8", "1.1.1.1", "203.0.113.7"]

    ip = rng.choice(internal_ips + external_ips)
    return {
        "req.user.role": rng.choice(roles),
        "req.ip": ip,
        "req.resource": rng.choice(resources),
        "req.method": rng.choice(methods),
    }


def _policy_templates(rng: Any) -> Tuple[str, Expr]:
    """Return (goal_text, target_expr_ast)."""
    # Choose a template; each produces a non-trivial policy.
    template_id = int(rng.randrange(3))

    if template_id == 0:
        # Role + internal IP + resource exclusion.
        goal = "Allow if user is admin, or if user is employee AND request is from an internal IP AND resource is not finance."
        expr = BinaryOp(
            "||",
            BinaryOp("==", Var("req.user.role"), StrLit("admin")),
            BinaryOp(
                "&&",
                BinaryOp("==", Var("req.user.role"), StrLit("employee")),
                BinaryOp(
                    "&&",
                    Call("is_internal_ip", (Var("req.ip"),)),
                    BinaryOp("!=", Var("req.resource"), StrLit("finance")),
                ),
            ),
        )
        return goal, expr

    if template_id == 1:
        # Public resources allowed, otherwise only admins or internal employees.
        goal = "Allow if resource starts with 'public/', otherwise allow only admins, or employees from internal IPs."
        expr = BinaryOp(
            "||",
            Call("starts_with", (Var("req.resource"), StrLit("public/"))),
            BinaryOp(
                "||",
                BinaryOp("==", Var("req.user.role"), StrLit("admin")),
                BinaryOp(
                    "&&",
                    BinaryOp("==", Var("req.user.role"), StrLit("employee")),
                    Call("is_internal_ip", (Var("req.ip"),)),
                ),
            ),
        )
        return goal, expr

    # template_id == 2
    goal = "Allow if method is GET and resource is public; or allow POST/DELETE only for admins on non-finance resources."
    is_public = Call("starts_with", (Var("req.resource"), StrLit("public/")))
    get_public = BinaryOp(
        "&&",
        BinaryOp("==", Var("req.method"), StrLit("GET")),
        is_public,
    )
    admin_write = BinaryOp(
        "&&",
        BinaryOp("==", Var("req.user.role"), StrLit("admin")),
        BinaryOp(
            "&&",
            BinaryOp("!=", Var("req.method"), StrLit("GET")),
            BinaryOp("!=", Var("req.resource"), StrLit("finance")),
        ),
    )
    return goal, BinaryOp("||", get_public, admin_write)


def _mutate_expr(rng: Any, expr: Expr) -> Expr:
    """Produce a *single* small bug in the expression."""

    def all_nodes(e: Expr) -> List[Expr]:
        out: List[Expr] = [e]
        if isinstance(e, UnaryOp):
            out.extend(all_nodes(e.expr))
        elif isinstance(e, BinaryOp):
            out.extend(all_nodes(e.left))
            out.extend(all_nodes(e.right))
        elif isinstance(e, Call):
            for a in e.args:
                out.extend(all_nodes(a))
        return out

    nodes = all_nodes(expr)
    # Prefer mutating something meaningful.
    candidates: List[Expr] = [n for n in nodes if isinstance(n, (BinaryOp, StrLit))]
    if not candidates:
        return expr

    target = rng.choice(candidates)

    # Helper: rebuild with a replacement of the exact node instance.
    def replace(e: Expr, old: Expr, new: Expr) -> Expr:
        if e is old:
            return new
        if isinstance(e, UnaryOp):
            return UnaryOp(e.op, replace(e.expr, old, new))
        if isinstance(e, BinaryOp):
            return BinaryOp(e.op, replace(e.left, old, new), replace(e.right, old, new))
        if isinstance(e, Call):
            return Call(e.name, tuple(replace(a, old, new) for a in e.args))
        return e

    if isinstance(target, BinaryOp):
        if target.op in ("&&", "||"):
            flipped = "||" if target.op == "&&" else "&&"
            return replace(expr, target, BinaryOp(flipped, target.left, target.right))
        if target.op in ("==", "!="):
            flipped = "!=" if target.op == "==" else "=="
            return replace(expr, target, BinaryOp(flipped, target.left, target.right))

    if isinstance(target, StrLit):
        pool = ["admin", "employee", "finance", "public/", "GET", "POST"]
        choice = rng.choice([p for p in pool if p != target.value] or pool)
        return replace(expr, target, StrLit(choice))

    return expr


def _make_cases(
    rng: Any,
    *,
    target_expr: Expr,
    builtins: Dict[str, BuiltinSig],
    n: int,
) -> List[Case]:
    cases: List[Case] = []
    # Ensure we get a mix of True/False.
    seen_true = 0
    seen_false = 0
    attempts = 0
    while len(cases) < n and attempts < 50 * n:
        attempts += 1
        inp = _sample_request(rng)
        got = bool(_eval_expr(target_expr, env=inp, builtins=builtins))
        if got:
            seen_true += 1
        else:
            seen_false += 1
        cases.append(Case(input=inp, expected=got))

    # If the generator accidentally produced an almost-constant policy, resample.
    if seen_true == 0 or seen_false == 0:
        # Force diversity by injecting two crafted cases.
        base = _sample_request(rng)
        base2 = dict(base)
        base2["req.user.role"] = "admin"
        base3 = dict(base)
        base3["req.user.role"] = "guest"
        got2 = bool(_eval_expr(target_expr, env=base2, builtins=builtins))
        got3 = bool(_eval_expr(target_expr, env=base3, builtins=builtins))
        cases = [Case(base2, got2), Case(base3, got3)] + cases[: max(0, n - 2)]
    return cases[:n]


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


class ExprPolicyEnv(HiddenSystemEnv):
    """Typed policy DSL debugging environment."""

    config = EXPRPOLICY_CONFIG

    @classmethod
    def domain_spec(
        cls,
        *,
        budget: int = 60,
        trap: bool = True,
        field_schema: Optional[Dict[str, str]] = None,
        builtin_sigs: Optional[List[str]] = None,
        operators: Optional[List[str]] = None,
        tool_costs: Optional[Dict[str, int]] = None,
        max_expr_len: int = 500,
        goal: Optional[str] = None,
        starter_expr: Optional[str] = None,
        public_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> DomainSpec:
        # Observation defaults
        if field_schema is None:
            field_schema = _default_field_schema()
        if operators is None:
            operators = ["&&", "||", "!", "==", "!=", "(", ")", ","]
        if builtin_sigs is None:
            builtin_sigs = [b.to_prompt_sig() for b in _default_builtins().values()]
        if tool_costs is None:
            tool_costs = {
                "type_check": 1,
                "run_tests_public": 5,
                "run_tests_hidden_preview": 6,
                "submit": int(cls.config.submission_cost),
            }

        return DomainSpec(
            actions=["type_check", "run_tests", "submit"],
            outputs=["ok", "error"],
            tool_schemas=[
                ToolSchema(
                    name="type_check",
                    description="Type-check a DSL expression. Returns Bool type on success, otherwise structured errors.",
                    args={
                        "expr": ArgSchema(
                            type="string",
                            description="DSL expression to type-check",
                        )
                    },
                    returns={
                        "ok": ReturnField("bool", "Whether expression type-checks"),
                        "type": ReturnField("string", "Type of expression if ok"),
                        "errors": ReturnField("list", "List of compile errors (if any)"),
                        "ast_nodes": ReturnField("int", "Approx AST size"),
                        "budget_left": ReturnField("int", "Remaining budget"),
                        "queries_used": ReturnField("int", "Total budget consumed"),
                        "trap_hit": ReturnField("bool", "Whether a trap was triggered"),
                    },
                ),
                ToolSchema(
                    name="run_tests",
                    description=(
                        "Run a deterministic test suite against an expression. "
                        "This simulates unit tests/CI. Suites have different costs."
                    ),
                    args={
                        "expr": ArgSchema(type="string", description="DSL expression to test"),
                        "suite": ArgSchema(
                            type="string",
                            enum=["public", "hidden_preview"],
                            description="Which suite to run",
                        ),
                    },
                    returns={
                        "ok": ReturnField("bool", "Whether tests ran (expression compiled)"),
                        "passed": ReturnField("bool", "Whether all suite cases passed"),
                        "suite": ReturnField("string", "Suite name"),
                        "num_cases": ReturnField("int", "Number of executed cases"),
                        "failed_case": ReturnField("object", "First failing case (if any)"),
                        "errors": ReturnField(
                            "list", "List of compile/type errors (if compilation fails)"
                        ),
                        "budget_left": ReturnField("int", "Remaining budget"),
                        "queries_used": ReturnField("int", "Total budget consumed"),
                        "trap_hit": ReturnField("bool", "Whether a trap was triggered"),
                    },
                ),
                ToolSchema(
                    name="submit",
                    description=(
                        "Submit a candidate expression for full hidden evaluation. "
                        "If correct, the episode ends successfully."
                    ),
                    args={
                        "expr": ArgSchema(type="string", description="Candidate DSL expression"),
                    },
                    returns={
                        "ok": ReturnField("bool", "Whether submission passes hidden tests"),
                        "counterexample": ReturnField(
                            "object", "Failing case from hidden suite (if feedback enabled)"
                        ),
                        "errors": ReturnField(
                            "list", "List of compile/type errors (if compilation fails)"
                        ),
                        "budget_left": ReturnField("int", "Remaining budget"),
                        "queries_used": ReturnField("int", "Total budget consumed"),
                        "trap_hit": ReturnField("bool", "Whether a trap was triggered"),
                    },
                ),
            ],
            hypothesis_schema={"type": "string"},
            observation_fields={
                "budget": ObservationField("int", "Query budget", budget),
                "trap": ObservationField("bool", "Whether traps exist", trap),
                "tool_costs": ObservationField("object", "Tool costs (budget units)", tool_costs),
                "max_expr_len": ObservationField(
                    "int", "Maximum expression length accepted", max_expr_len
                ),
                "field_schema": ObservationField(
                    "object", "Available request fields and their types", field_schema
                ),
                "operators": ObservationField(
                    "list", "Available operators and punctuation", operators
                ),
                "builtins": ObservationField(
                    "list", "Available built-in functions (signatures)", builtin_sigs
                ),
                "goal": ObservationField("string", "Natural language policy objective", goal or ""),
                "starter_expr": ObservationField(
                    "string", "Buggy starting expression", starter_expr or ""
                ),
                "public_cases": ObservationField(
                    "list", "Public example test cases", public_cases or []
                ),
            },
            params={
                "max_expr_len": ParamSpec(
                    type="int",
                    description="Maximum allowed expression length (characters)",
                    default=max_expr_len,
                    bounds=(50, 2000),
                ),
                "n_public": ParamSpec(
                    type="int",
                    description="Number of public (visible) test cases",
                    default=8,
                    bounds=(2, 30),
                ),
                "n_hidden": ParamSpec(
                    type="int",
                    description="Number of hidden test cases",
                    default=80,
                    bounds=(10, 500),
                ),
            },
            skin_name="exprpolicy",
            n_states=1,
            has_traps=trap,
        )

    @classmethod
    def domain_params_from_answer(cls, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        dsl = answer_data.get("dsl", {})
        return {
            "field_schema": dsl.get("field_schema"),
            "builtin_sigs": dsl.get("builtin_sigs"),
            "operators": dsl.get("operators"),
            "tool_costs": dsl.get("tool_costs"),
            "max_expr_len": dsl.get("max_expr_len"),
            "goal": answer_data.get("goal"),
            "starter_expr": answer_data.get("starter_expr"),
            "public_cases": answer_data.get("public_cases"),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._field_schema: Dict[str, TypeName] = {}
        self._builtins: Dict[str, BuiltinSig] = {}
        self._public_cases: List[Case] = []
        self._hidden_cases: List[Case] = []
        self._hidden_preview: List[Case] = []
        self._tool_costs: Dict[str, int] = {}
        self._max_expr_len: int = 500
        self._banned_tokens: List[str] = []

    # ─────────────────────────────────────────────────────────────
    # Required abstract implementations
    # ─────────────────────────────────────────────────────────────

    def _configure_from_metadata(self, meta: Dict[str, Any]) -> None:
        dsl = meta.get("dsl", {})
        raw_schema = dsl.get("field_schema") or _default_field_schema()
        self._field_schema = {str(k): str(v) for k, v in dict(raw_schema).items()}

        # Builtins are kept fixed in v1; the episode can still list them explicitly.
        self._builtins = _default_builtins()
        self._tool_costs = dict(dsl.get("tool_costs") or {})
        if not self._tool_costs:
            self._tool_costs = {
                "type_check": 1,
                "run_tests_public": 5,
                "run_tests_hidden_preview": 6,
                "submit": int(self.config.submission_cost),
            }

        self._max_expr_len = int(dsl.get("max_expr_len", 500))
        self._banned_tokens = list(dsl.get("banned_tokens") or [])

        self._public_cases = [
            Case(input=dict(c.get("input", {})), expected=bool(c.get("expected", False)))
            for c in (meta.get("public_cases") or [])
        ]
        self._hidden_cases = [
            Case(input=dict(c.get("input", {})), expected=bool(c.get("expected", False)))
            for c in (meta.get("hidden_cases") or [])
        ]
        self._hidden_preview = [
            Case(input=dict(c.get("input", {})), expected=bool(c.get("expected", False)))
            for c in (meta.get("hidden_preview") or [])
        ]

        # Ground truth is the hidden test suite. (We don't need the target expression at runtime.)
        self._ground_truth = {
            "hidden_cases": [c.to_dict() for c in self._hidden_cases],
        }

    def _get_start_state(self) -> int:
        return 0

    def _get_tools(self) -> List:
        return [self.type_check, self.run_tests, self.submit]

    # ─────────────────────────────────────────────────────────────
    # Tools
    # ─────────────────────────────────────────────────────────────

    def _check_common(self, cost: int) -> Optional[str]:
        state = self._state()
        if state["done"]:
            return self._episode_finished()
        if int(state.get("budget", 0)) <= 0:
            return self._budget_exhausted()
        if not self._consume_budget(int(max(0, cost))):
            return self._budget_exhausted()
        state["steps"] = int(state.get("steps", 0)) + 1
        return None

    def _trap_check(self, expr: str) -> bool:
        state = self._state()
        if not bool(state.get("trap_hit", False)) and self._banned_tokens:
            for tok in self._banned_tokens:
                if not tok:
                    continue
                # Avoid substring false positives (e.g., "executive" contains "exec").
                if re.search(rf"\b{re.escape(tok)}\b", expr, flags=re.IGNORECASE):
                    self._mark_trap_hit()
                    return True
        return bool(state.get("trap_hit", False))

    def type_check(self, expr: str) -> str:
        """Compile (parse + type-check) an expression."""
        cost = int(self._tool_costs.get("type_check", 1))
        err = self._check_common(cost)
        if err is not None:
            return err

        if not isinstance(expr, str):
            return self._tool_error(error_invalid_argument("expr must be a string"))

        expr = expr.strip()
        self._trap_check(expr)
        if len(expr) > self._max_expr_len:
            errors = [
                CompileError(
                    "ParseError",
                    f"Expression too long (len={len(expr)} > max={self._max_expr_len})",
                )
            ]
            return json.dumps(
                {
                    "ok": False,
                    "type": "",
                    "errors": [e.to_dict() for e in errors],
                    "ast_nodes": 0,
                    "budget_left": int(self._state().get("budget", 0)),
                    "queries_used": int(self._state().get("queries_used", 0)),
                    "trap_hit": bool(self._state().get("trap_hit", False)),
                }
            )

        ast, t, errors = _compile(expr, field_schema=self._field_schema, builtins=self._builtins)
        return json.dumps(
            {
                "ok": t is not None and not errors,
                "type": t or "",
                "errors": [e.to_dict() for e in errors],
                "ast_nodes": _ast_size(ast) if ast is not None else 0,
                "budget_left": int(self._state().get("budget", 0)),
                "queries_used": int(self._state().get("queries_used", 0)),
                "trap_hit": bool(self._state().get("trap_hit", False)),
            }
        )

    def run_tests(self, expr: str, suite: str) -> str:
        """Run public or hidden-preview tests."""
        suite = str(suite)
        if suite == "public":
            cost = int(self._tool_costs.get("run_tests_public", 5))
            cases = self._public_cases
        elif suite == "hidden_preview":
            cost = int(self._tool_costs.get("run_tests_hidden_preview", 6))
            cases = self._hidden_preview
        else:
            # Consume cost of a bad call as if it were public tests.
            cost = int(self._tool_costs.get("run_tests_public", 5))
            err = self._check_common(cost)
            if err is not None:
                return err
            return self._tool_error(
                error_invalid_argument(
                    f"Invalid suite '{suite}'. Must be one of: ['public','hidden_preview']",
                    details={"received": suite, "valid": ["public", "hidden_preview"]},
                )
            )

        err = self._check_common(cost)
        if err is not None:
            return err

        if not isinstance(expr, str):
            return self._tool_error(error_invalid_argument("expr must be a string"))

        expr = expr.strip()
        self._trap_check(expr)

        if len(expr) > self._max_expr_len:
            errors = [
                CompileError(
                    "ParseError",
                    f"Expression too long (len={len(expr)} > max={self._max_expr_len})",
                )
            ]
            return json.dumps(
                {
                    "ok": False,
                    "passed": False,
                    "suite": suite,
                    "num_cases": int(len(cases)),
                    "failed_case": None,
                    "errors": [e.to_dict() for e in errors],
                    "budget_left": int(self._state().get("budget", 0)),
                    "queries_used": int(self._state().get("queries_used", 0)),
                    "trap_hit": bool(self._state().get("trap_hit", False)),
                }
            )

        ast, t, errors = _compile(expr, field_schema=self._field_schema, builtins=self._builtins)
        if t is None or errors:
            return json.dumps(
                {
                    "ok": False,
                    "passed": False,
                    "suite": suite,
                    "num_cases": int(len(cases)),
                    "failed_case": None,
                    "errors": [e.to_dict() for e in errors],
                    "budget_left": int(self._state().get("budget", 0)),
                    "queries_used": int(self._state().get("queries_used", 0)),
                    "trap_hit": bool(self._state().get("trap_hit", False)),
                }
            )

        if t != T_BOOL:
            # Policy expressions must return Bool.
            ce = CompileError(
                "TypeMismatch",
                "Policy expression must have type Bool",
                expected=T_BOOL,
                found=t,
            )
            return json.dumps(
                {
                    "ok": False,
                    "passed": False,
                    "suite": suite,
                    "num_cases": int(len(cases)),
                    "failed_case": None,
                    "errors": [ce.to_dict()],
                    "budget_left": int(self._state().get("budget", 0)),
                    "queries_used": int(self._state().get("queries_used", 0)),
                    "trap_hit": bool(self._state().get("trap_hit", False)),
                }
            )

        compiled_ast = cast(Expr, ast)
        failed: Optional[Dict[str, Any]] = None
        for idx, c in enumerate(cases):
            got = bool(_eval_expr(compiled_ast, env=c.input, builtins=self._builtins))
            if got != bool(c.expected):
                failed = {
                    "case_index": idx,
                    "input": c.input,
                    "expected": bool(c.expected),
                    "got": bool(got),
                }
                break

        return json.dumps(
            {
                "ok": True,
                "passed": failed is None,
                "suite": suite,
                "num_cases": int(len(cases)),
                "failed_case": failed,
                "errors": [],
                "budget_left": int(self._state().get("budget", 0)),
                "queries_used": int(self._state().get("queries_used", 0)),
                "trap_hit": bool(self._state().get("trap_hit", False)),
            }
        )

    def submit(self, expr: str) -> str:
        """Submit an expression for full hidden evaluation."""
        cost = int(self._tool_costs.get("submit", self.config.submission_cost))
        err = self._check_common(cost)
        if err is not None:
            return err

        if not isinstance(expr, str):
            return self._tool_error(error_invalid_argument("expr must be a string"))

        expr = expr.strip()
        self._trap_check(expr)

        if len(expr) > self._max_expr_len:
            errors = [
                CompileError(
                    "ParseError",
                    f"Expression too long (len={len(expr)} > max={self._max_expr_len})",
                )
            ]
            return json.dumps(
                {
                    "ok": False,
                    "counterexample": None,
                    "errors": [e.to_dict() for e in errors],
                    "budget_left": int(self._state().get("budget", 0)),
                    "queries_used": int(self._state().get("queries_used", 0)),
                    "trap_hit": bool(self._state().get("trap_hit", False)),
                }
            )

        ast, t, errors = _compile(expr, field_schema=self._field_schema, builtins=self._builtins)
        if t is None or errors or t != T_BOOL:
            # Submission can continue; it's just incorrect.
            return json.dumps(
                {
                    "ok": False,
                    "counterexample": None,
                    "errors": [e.to_dict() for e in errors]
                    + (
                        []
                        if t in (None, T_BOOL)
                        else [
                            CompileError(
                                "TypeMismatch",
                                "Policy expression must have type Bool",
                                expected=T_BOOL,
                                found=t,
                            ).to_dict()
                        ]
                    ),
                    "budget_left": int(self._state().get("budget", 0)),
                    "queries_used": int(self._state().get("queries_used", 0)),
                    "trap_hit": bool(self._state().get("trap_hit", False)),
                }
            )

        compiled_ast = cast(Expr, ast)
        cex: Optional[Dict[str, Any]] = None
        ok = True
        for idx, c in enumerate(self._hidden_cases):
            got = bool(_eval_expr(compiled_ast, env=c.input, builtins=self._builtins))
            if got != bool(c.expected):
                ok = False
                if self.feedback_enabled:
                    cex = {
                        "case_index": idx,
                        "input": c.input,
                        "expected": bool(c.expected),
                        "got": bool(got),
                    }
                break

        if ok:
            self._end_episode(success=True)

        return json.dumps(
            {
                "ok": bool(ok) and not bool(self._state().get("trap_hit", False)),
                "counterexample": cex,
                "errors": [],
                "budget_left": int(self._state().get("budget", 0)),
                "queries_used": int(self._state().get("queries_used", 0)),
                "trap_hit": bool(self._state().get("trap_hit", False)),
            }
        )

    # ─────────────────────────────────────────────────────────────
    # Prompting
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def get_prompt_template(
        cls, obs: Dict[str, Any], *, feedback: bool = False
    ) -> List[Dict[str, str]]:
        budget = int(obs.get("budget", cls.config.default_budget))
        trap = bool(obs.get("trap", True))

        spec_kwargs: Dict[str, Any] = {
            "budget": budget,
            "trap": trap,
            "max_expr_len": int(obs.get("max_expr_len", 500)),
        }
        if obs.get("field_schema") is not None:
            spec_kwargs["field_schema"] = obs.get("field_schema")
        if obs.get("operators") is not None:
            spec_kwargs["operators"] = obs.get("operators")
        if obs.get("builtins") is not None:
            spec_kwargs["builtin_sigs"] = obs.get("builtins")
        if obs.get("tool_costs") is not None:
            spec_kwargs["tool_costs"] = obs.get("tool_costs")
        if obs.get("goal") is not None:
            spec_kwargs["goal"] = obs.get("goal")
        if obs.get("starter_expr") is not None:
            spec_kwargs["starter_expr"] = obs.get("starter_expr")
        if obs.get("public_cases") is not None:
            spec_kwargs["public_cases"] = obs.get("public_cases")

        spec = cls.domain_spec(**spec_kwargs)
        tools_text = spec.format_tools_for_prompt()

        sys_msg = {
            "role": "system",
            "content": (
                "You are an autonomous tool-using agent working with a typed policy DSL.\n"
                "Your job is to fix a buggy policy expression so it passes hidden tests (like CI).\n"
                "Return ONLY function tool calls; never output natural language.\n\n"
                "Environment semantics:\n"
                "- type_check(expr) simulates compiler/type-checker feedback (rich diagnostics).\n"
                "- run_tests(expr, suite) simulates running deterministic tests (public / hidden preview).\n"
                "- submit(expr) runs full hidden evaluation. If correct, episode ends successfully.\n"
                "- Each tool consumes budget according to tool_costs in OBSERVATION.\n"
                + ("- Traps exist: banned tokens immediately fail the episode.\n" if trap else "")
                + (
                    "- If submission is wrong, a counterexample will be returned.\n"
                    if feedback
                    else ""
                )
                + "\nTools:\n"
                + tools_text
                + "\n\nRespond only with tool calls."
            ),
        }

        obs_values = {
            k: v for k, v in obs.items() if k in spec.observation_fields and v is not None
        }
        obs_values["budget"] = budget
        obs_values["trap"] = trap
        obs_json = spec.build_observation(**obs_values)

        usr_msg = {
            "role": "user",
            "content": (
                "OBSERVATION:\n" + json.dumps(obs_json) + "\n\nRespond only with tool calls."
            ),
        }

        return [sys_msg, usr_msg]

    # ─────────────────────────────────────────────────────────────
    # Static generation
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def generate_system_static(
        seed: int,
        *,
        n_public: int = 8,
        n_hidden: int = 80,
        max_expr_len: int = 500,
        trap: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        rng = get_rng(seed)
        field_schema = _default_field_schema()
        builtins = _default_builtins()

        goal, target_ast = _policy_templates(rng)
        target_expr = _expr_to_str(target_ast)

        # Generate test cases from target.
        hidden_cases = _make_cases(rng, target_expr=target_ast, builtins=builtins, n=int(n_hidden))
        public_cases = hidden_cases[: int(n_public)]

        # Create a buggy starter expr that fails at least one public case.
        starter_expr = target_expr
        for _ in range(30):
            cand_ast = _mutate_expr(rng, target_ast)
            cand_expr = _expr_to_str(cand_ast)
            if len(cand_expr) > int(max_expr_len):
                continue
            # Evaluate on public cases.
            ok = True
            for c in public_cases:
                got = bool(_eval_expr(cand_ast, env=c.input, builtins=builtins))
                if got != bool(c.expected):
                    ok = False
                    break
            if not ok:
                starter_expr = cand_expr
                break

        # Guarantee starter differs behaviorally on at least one public case.
        if starter_expr == target_expr and public_cases:
            neg = f"!({target_expr})"
            if len(neg) <= int(max_expr_len):
                starter_expr = neg
            else:
                # Always fails at least the first case.
                starter_expr = "true" if not bool(public_cases[0].expected) else "false"

        hidden_preview = hidden_cases[int(n_public) : int(n_public) + 5]

        tool_costs = {
            "type_check": 1,
            "run_tests_public": 5,
            "run_tests_hidden_preview": 6,
            "submit": int(EXPRPOLICY_CONFIG.submission_cost),
        }

        dsl = {
            "field_schema": field_schema,
            "builtin_sigs": [b.to_prompt_sig() for b in builtins.values()],
            "operators": ["&&", "||", "!", "==", "!=", "(", ")", ","],
            "tool_costs": tool_costs,
            "max_expr_len": int(max_expr_len),
            "banned_tokens": ["eval", "exec", "import"] if trap else [],
        }

        return {
            "dsl": dsl,
            "goal": goal,
            "starter_expr": starter_expr,
            # Store target expr for debugging / unit tests (not included in observation).
            "target_expr": target_expr,
            "public_cases": [c.to_dict() for c in public_cases],
            "hidden_preview": [c.to_dict() for c in hidden_preview],
            "hidden_cases": [c.to_dict() for c in hidden_cases],
        }
