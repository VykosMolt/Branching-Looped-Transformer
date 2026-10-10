# Note: duplicate Qwen launch, 2026-10-10 20:50 (coordinator error)

The agent's `run_full.sh` (PID 2378415) runs DeBERTa then Qwen. After that agent stopped at a usage limit,
the coordinator (Claude, main session) did not notice the chaining and started a second
`lm_reader.py --reader qwen > runs_qwen.log` at 20:50:06, 27 s after the real Qwen run (PID 2427107) began
at 20:49:39. The duplicate hit CUDA out-of-memory on its first fit and exited after 10 s (exit 1).
It wrote no fit results (runs/qwen/ held no result files when checked at 20:51).

Effects, left as they are (logs are not rewritten):
- `runs_qwen.log` was truncated by the duplicate's redirect: the real run's first ~27 s of output (model loading)
  is lost, and the duplicate's traceback sits in the file. The real run keeps writing at its own offset.
- `run_times.log` contains two extra lines from the duplicate: "deberta end (observed) 20:50:06",
  "qwen start 20:50:06" and "qwen end 20:50:16 exit 1". The real run's lines are "deberta end 20:49:39 exit 0"
  and "qwen start 20:49:39".
The real run's per-fit JSON files under runs/qwen/ are the record of results; they are unaffected.
