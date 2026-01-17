"""dedeucerl-generate: CLI for generating DedeuceRL task splits."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any, Dict, List, Optional

from dedeucerl.skins import SKIN_REGISTRY
from dedeucerl.core import TaskGenerator


def parse_seeds(seeds_str: str) -> List[int]:
    """Parse seed specification like '0-100' or '0,1,2,3'."""
    seeds = []
    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            # Range like "0-100"
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_args(argv: Optional[List[str]] = None) -> tuple[argparse.Namespace, List[str]]:
    """Parse command-line arguments.

    Uses a two-phase parse so skins can expose parameters without hardcoding.
    Unknown args are parsed later using the skin's `DomainSpec.params` registry.
    """
    parser = argparse.ArgumentParser(
        prog="dedeucerl-generate",
        description="Generate DedeuceRL task splits for evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 Mealy tasks
  dedeucerl-generate --skin mealy --seeds 0-99 --budget 25 --n-states 3 -o seeds/mealy_dev.json

  # Generate Protocol tasks with custom endpoints
  dedeucerl-generate --skin protocol --seeds 0-49 --budget 30 --n-endpoints 4 --n-states 3
        """,
    )

    parser.add_argument(
        "--skin",
        required=True,
        choices=list(SKIN_REGISTRY.keys()),
        help="Skin to generate tasks for.",
    )
    parser.add_argument(
        "--show-skin-params",
        action="store_true",
        help="Print available skin parameters and exit.",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        type=str,
        help="Seed specification: '0-99' for range, or '0,1,2,3' for list.",
    )
    parser.add_argument(
        "--subset",
        default="dev",
        type=str,
        help="Subset name (default: 'dev').",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=25,
        help="Query budget per episode (default: 25).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        type=str,
        help="Output JSON file (default: seeds/<skin>_<subset>.json).",
    )
    parser.add_argument(
        "--no-trap",
        action="store_true",
        help="Disable trap transitions.",
    )

    # Generic skin kwargs
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Additional skin parameter as KEY=VALUE (repeatable).",
    )
    parser.add_argument(
        "--skin-kwargs",
        type=str,
        default=None,
        help="Additional skin kwargs as a JSON object string.",
    )

    return parser.parse_known_args(argv)


def main():
    """Entry point for dedeucerl-generate CLI."""
    args, unknown = parse_args()

    # Get skin class
    if args.skin not in SKIN_REGISTRY:
        print(f"Error: Unknown skin '{args.skin}'", file=sys.stderr)
        print(f"Available: {list(SKIN_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    SkinClass = SKIN_REGISTRY[args.skin]
    generator = TaskGenerator(SkinClass)

    if args.show_skin_params:
        try:
            spec = SkinClass.domain_spec(budget=int(args.budget), trap=not args.no_trap)
        except TypeError:
            spec = SkinClass.domain_spec()
        param_specs = getattr(spec, "params", {}) or {}
        if not param_specs:
            print(f"Skin '{args.skin}' does not declare any parameters.")
            sys.exit(0)

        print(f"Skin '{args.skin}' parameters:")
        for name in sorted(param_specs.keys()):
            ps = param_specs[name]
            default = getattr(ps, "default", None)
            desc = getattr(ps, "description", "")
            print(f"- {name} (default={default}): {desc}")
        sys.exit(0)

    # Parse seeds
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as e:
        print(f"Error parsing seeds: {e}", file=sys.stderr)
        sys.exit(1)

    def _parse_value(raw: str) -> Any:
        v = raw.strip()
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        # Allow JSON literals/objects/arrays.
        try:
            return json.loads(v)
        except Exception:
            return v

    # Resolve supported skin parameters from the skin itself
    try:
        spec = SkinClass.domain_spec(budget=int(args.budget), trap=not args.no_trap)
    except TypeError:
        # Some skins may not accept budget/trap in domain_spec
        spec = SkinClass.domain_spec()

    param_specs = getattr(spec, "params", {}) or {}

    def _coerce_param(name: str, raw: str) -> Any:
        ps = param_specs[name]
        t = getattr(ps, "type", "json")
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        if t == "bool":
            v = raw.strip().lower()
            if v in ("true", "1", "yes", "y"):
                return True
            if v in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"Invalid bool for {name}: {raw}")
        if t == "str":
            return str(raw)
        # json
        return json.loads(raw)

    def _parse_unknown_params(tokens: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not tok.startswith("--"):
                raise ValueError(f"Unexpected token: {tok}")

            keypart = tok[2:]
            if "=" in keypart:
                key_raw, value_raw = keypart.split("=", 1)
                key = key_raw.replace("-", "_")
                if key not in param_specs:
                    raise ValueError(f"Unknown parameter: --{key_raw}")
                out[key] = _coerce_param(key, value_raw)
                i += 1
                continue

            # Support --no-foo for boolean params
            if keypart.startswith("no-") or keypart.startswith("no_"):
                key = keypart[3:].replace("-", "_")
                if key not in param_specs:
                    raise ValueError(f"Unknown parameter: --{keypart}")
                ps_type = getattr(param_specs[key], "type", None)
                if ps_type != "bool":
                    raise ValueError(f"--{keypart} only valid for bool params")
                out[key] = False
                i += 1
                continue

            key = keypart.replace("-", "_")
            if key not in param_specs:
                raise ValueError(f"Unknown parameter: --{keypart}")

            ps_type = getattr(param_specs[key], "type", None)
            if ps_type == "bool":
                # Allow bare --flag (sets True)
                if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                    out[key] = True
                    i += 1
                    continue

            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value for --{keypart}")
            out[key] = _coerce_param(key, tokens[i + 1])
            i += 2

        return out

    # Build skin kwargs
    skin_kwargs: Dict[str, Any] = {"trap": not args.no_trap}

    # Parse `--foo 3` style params from unknown args, if skin declares them.
    if unknown:
        if not param_specs:
            print(
                f"Error: Skin '{args.skin}' does not declare extra params; use --param/--skin-kwargs.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            skin_kwargs.update(_parse_unknown_params(unknown))
        except Exception as e:
            available = ", ".join(sorted(param_specs.keys()))
            print(f"Error parsing skin parameters: {e}", file=sys.stderr)
            print(f"Available params for '{args.skin}': {available}", file=sys.stderr)
            print(
                "Tip: you can also pass --param KEY=VALUE or --skin-kwargs '{...}'", file=sys.stderr
            )
            sys.exit(1)

    # Apply defaults from the param registry
    for name, ps in param_specs.items():
        if name not in skin_kwargs and getattr(ps, "default", None) is not None:
            skin_kwargs[name] = ps.default

    # Allow power-user overrides
    if args.skin_kwargs:
        try:
            extra = json.loads(args.skin_kwargs)
            if not isinstance(extra, dict):
                raise ValueError("--skin-kwargs must be a JSON object")
            skin_kwargs.update(extra)
        except Exception as e:
            print(f"Error parsing --skin-kwargs: {e}", file=sys.stderr)
            sys.exit(1)

    for kv in args.param:
        if "=" not in kv:
            print(f"Error: --param must be KEY=VALUE, got: {kv}", file=sys.stderr)
            sys.exit(1)
        key, value = kv.split("=", 1)
        skin_kwargs[key.strip()] = _parse_value(value)

    # Filter kwargs if generate_system_static is strict
    gen_sig = inspect.signature(SkinClass.generate_system_static)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in gen_sig.parameters.values()
    )
    if not accepts_var_kw:
        allowed = set(gen_sig.parameters.keys())
        allowed.discard("seed")
        skin_kwargs = {k: v for k, v in skin_kwargs.items() if k in allowed}

    # Generate split
    print(f"Generating {len(seeds)} episodes for skin '{args.skin}'...")
    print(f"  Budget: {args.budget}")
    print(f"  Params: {skin_kwargs}")

    split = generator.generate_split(
        seeds=seeds,
        budget=args.budget,
        subset_name=args.subset,
        **skin_kwargs,
    )

    # Determine output path
    out_path = args.out
    if out_path is None:
        out_path = f"seeds/{args.skin}_{args.subset}.json"

    # Save
    generator.save_split(split, out_path)

    print(f"\nSaved to: {out_path}")
    print(f"Episodes: {len(split[args.subset]['items'])}")

    # Show sample
    print("\nSample item (seed 0):")
    sample = split[args.subset]["items"][0]
    print(json.dumps(sample, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
