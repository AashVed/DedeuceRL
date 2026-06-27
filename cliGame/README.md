# DedeuceRL Interactive Game

Run:

```bash
python -m cliGame
```

The game lists registered kernels, samples one task instance, prints the prompt
and tools, then lets you call tools manually. It uses `EpisodeRuntime` directly,
so behavior matches CLI eval and Verifiers surfaces.

Examples:

```text
act A
act {"symbol":"A"}
submit_table {"n":2,"start":0,"trans":{...}}
```

Commands: `:help`, `:prompt`, `:state`, `:quit`.
