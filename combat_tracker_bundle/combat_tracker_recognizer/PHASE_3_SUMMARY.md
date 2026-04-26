# Phase 3 summary

End-of-session review with HDBSCAN clustering, cmd-based REPL CLI, and
the human-in-the-loop label/discard/merge/split/relabel/commit/rollback
loop. Labels flow back into the bank; replayed events route KNOWN.

## Built

- `review/cluster.py` — `cluster_unknowns(clips, parent, config, bank=)`
  HDBSCAN on L2-normalised embeddings (euclidean is order-equivalent to
  cosine on unit vectors). Real clusters and noise points are both
  returned; noise points become singletons one-per-clip so each can be
  labeled or discarded individually. Suggested labels per cluster come
  from `bank.match` on the medoid.
- `review/session.py` — `ReviewSession` with `list_clusters`,
  `get_cluster`, `label_cluster`, `discard_cluster`, `relabel_clip`,
  `merge_clusters`, `split_clip_out`, `uncommitted_changes`, `commit`,
  `rollback`. An entry snapshot is taken at session open so rollback
  always has a target. `commit` writes a fresh snapshot tagged with the
  session id and optional note. Bank-mutating ops auto-snapshot before
  destructive paths.
- `review/cli.py` — `cmd.Cmd`-based REPL with `list`, `show`, `play`
  (uses `ffplay` if available; prints fallback path if not), `label`,
  `relabel`, `discard`, `split`, `merge`, `status`, `commit`,
  `rollback`, `quit`. Plain-text table rendering, no extra deps.
- `__main__.py` — entry point so `python -m combat_tracker_recognizer
  review <session_id>` works exactly as the plan specifies.

## Tests

`pytest combat_tracker_recognizer/tests/`: **48 passed, 1 xpass**.

Combined with the parent tracker tests:

`pytest tests/tracking/ combat_tracker_recognizer/tests/`: **96 passed,
1 xpass**.

New tests:
- `test_cluster.py` (5 tests): three-cluster recovery, noise-point
  singletons, medoid is in member set, suggested labels come from the
  bank, parent filtering.
- `test_review_session.py` (7 tests): label_cluster mutates bank +
  store, commit creates a snapshot row, rollback restores bank state,
  discard removes from unlabeled, merge combines membership, split
  creates singleton, full Phase 3 e2e (label → commit → replay →
  KNOWN).

## Smoke test

`test_phase3_e2e_label_then_replay_routes_known` in
`test_review_session.py` is the Phase 3 smoke test. It walks the entire
loop: push two synthetic jab events into the unlabeled set, open a
ReviewSession, label whatever clusters HDBSCAN produces as `jab`,
commit, then replay a fresh jab event with a different seed and assert
it routes KNOWN as `jab`.

## CLI manual run (requires built DB)

```
python -m combat_tracker_recognizer review my_session \
    --db ./recognizer.db --parent punch
```

Inside the shell:

```
list                          # see clusters
show 0                        # inspect cluster 0 in detail
play 0                        # ffplay the exemplar (or skip if not installed)
label 0 jab                   # apply 'jab' to every clip in cluster 0
status                        # see uncommitted changes
commit --note "session 1"     # persist + snapshot
quit
```

## Known limitations

- HDBSCAN is sensitive to `min_cluster_size` for small N. The smoke
  test sets `min_cluster_size=2` to make 2-clip sets clusterable; in
  production the default of 3 is more sensible but requires more
  data per session.
- `play --all` runs ffplay sequentially per clip; no parallelism.
- `merge_clusters` operates only on the in-memory cluster cache; on
  the next `list_clusters` call HDBSCAN re-runs and may re-split.
  This is fine because labels persist independently of the cluster
  view.
- The CLI is single-user; no concurrent access protection on the
  ClipStore (sqlite handles row-level locking but the bank in-process
  state isn't shared).

## Cross-cutting plan items still TODO (out of phase scope)

- `mypy --strict` on public modules (cross-cutting constraint #3).
  Type hints exist but I didn't run mypy in this loop.
- > 80% line coverage report (cross-cutting constraint in testing
  requirements). Not measured in this run; can be added with
  `pytest-cov`.
- Logging configured with named loggers per submodule. Library code
  has no `print()` calls (CLI does, which is correct), but logger
  setup hasn't been wired in.

## Done

The plan's three phases are complete. Branch:
`claude/boxing-identity-tracking-BuRRr`. All 96 tests across both the
parent `tracking/` package and the new `combat_tracker_recognizer/`
package pass.
