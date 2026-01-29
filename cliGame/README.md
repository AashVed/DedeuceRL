# DedeuceRL CLI Game

Play any DedeuceRL skin interactively as a human "agent".

Run:

```bash
python -m cliGame
```

How it works:
- You pick a skin from `dedeucerl.skins.SKIN_REGISTRY`.
- You enter a seed.
- The game uses the same generation parameters as `HF_DATA/GENERATION.json` (test subset).
- It prints the exact system/user prompt.
- You enter tool calls and see tool JSON outputs.

Input examples:

```text
act A
act {"symbol":"A"}
submit_table {"n":4,"start":0,"trans":{...}}
type_check true
run_tests {"expr":"true","suite":"public"}
```

Commands:
- `:help` `:tools` `:prompt` `:state` `:quit`
