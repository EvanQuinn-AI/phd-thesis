"""REPL CLI: ``python -m combat_tracker_recognizer review <session_id>``.

Uses ``cmd.Cmd`` from stdlib. No external dependencies beyond ffplay
(optional, fails gracefully).
"""

from __future__ import annotations

import argparse
import cmd
import shlex
import shutil
import subprocess
import sys
from typing import Optional

from combat_tracker_recognizer.config import RecognizerConfig
from combat_tracker_recognizer.recognizer import SubclassActionRecognizer
from combat_tracker_recognizer.review.session import ReviewSession


def _format_table(rows: list[list[str]], header: list[str]) -> str:
    cols = list(zip(*([header] + rows))) if rows else [[h] for h in header]
    widths = [max(len(str(x)) for x in col) for col in cols]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "  ".join("-" * w for w in widths)
    out = [line, sep]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


class ReviewShell(cmd.Cmd):
    intro = ("Recognizer review shell. Type 'help' for commands, 'quit' to exit.\n"
             "Always 'commit' before quitting to persist label/discard work.")
    prompt = "(review) "

    def __init__(self, session: ReviewSession, parent_filter: Optional[str] = None):
        super().__init__()
        self.session = session
        self.parent_filter = parent_filter

    # ---- Commands ------------------------------------------------------

    def do_list(self, arg: str) -> None:
        """list [parent] — show clusters with optional parent filter."""
        parent = arg.strip() or self.parent_filter
        clusters = self.session.list_clusters(parent=parent)
        if not clusters:
            print("(no unlabeled clips)")
            return
        rows = []
        for c in clusters:
            sugg = ", ".join(f"{s}:{d:.2f}" for s, d in c.suggested_labels[:3]) or "-"
            rows.append([str(c.id), c.parent_class, str(c.size),
                         f"{c.intra_distance_mean:.3f}", sugg])
        print(_format_table(rows, ["id", "parent", "size", "intra_d", "suggested"]))

    def do_show(self, arg: str) -> None:
        """show <cluster_id>"""
        try:
            cid = int(arg.strip())
        except ValueError:
            print("usage: show <cluster_id>"); return
        c = self.session.get_cluster(cid)
        print(f"Cluster {c.id} parent={c.parent_class} size={c.size}")
        print(f"  exemplar clip_id={c.exemplar_clip_id}")
        print(f"  members: {c.member_clip_ids}")
        print("  suggested labels:")
        if c.suggested_labels:
            for sc, d in c.suggested_labels:
                print(f"    {sc:24s}  d={d:.4f}")
        else:
            print("    (none — bank has no entries for this parent)")

    def do_play(self, arg: str) -> None:
        """play <cluster_id> [--all] — invoke ffplay on the exemplar (or all members)."""
        parts = shlex.split(arg)
        if not parts:
            print("usage: play <cluster_id> [--all]"); return
        try:
            cid = int(parts[0])
        except ValueError:
            print("usage: play <cluster_id> [--all]"); return
        play_all = "--all" in parts[1:]
        c = self.session.get_cluster(cid)
        clip_ids = c.member_clip_ids if play_all else [c.exemplar_clip_id]
        if shutil.which("ffplay") is None:
            print("ffplay is not installed. Open these clips manually:")
            for clip_id in clip_ids:
                clip = self.session.store.get_clip(clip_id)
                print(f"  clip {clip_id}: {clip.video_ref}  "
                      f"frames {clip.pose.frame_start}–{clip.pose.frame_end}")
            return
        for clip_id in clip_ids:
            clip = self.session.store.get_clip(clip_id)
            if not clip.video_ref:
                print(f"clip {clip_id} has no video_ref; skipping")
                continue
            start_t = clip.pose.frame_start / max(clip.pose.fps, 1.0)
            duration = (clip.pose.frame_end - clip.pose.frame_start + 1) / max(clip.pose.fps, 1.0)
            subprocess.run([
                "ffplay", "-ss", f"{start_t:.3f}", "-t", f"{duration:.3f}",
                "-autoexit", clip.video_ref,
            ], check=False)

    def do_label(self, arg: str) -> None:
        """label <cluster_id> <subclass> [--parent <parent>]"""
        parts = shlex.split(arg)
        if len(parts) < 2:
            print("usage: label <cluster_id> <subclass> [--parent <parent>]"); return
        try:
            cid = int(parts[0])
        except ValueError:
            print("first arg must be a cluster id"); return
        subclass = parts[1]
        parent = None
        if "--parent" in parts:
            i = parts.index("--parent")
            parent = parts[i + 1] if i + 1 < len(parts) else None
        self.session.label_cluster(cid, subclass, parent_class=parent)
        print(f"labeled cluster {cid} as {subclass}")

    def do_relabel(self, arg: str) -> None:
        """relabel <clip_id> <subclass>"""
        parts = shlex.split(arg)
        if len(parts) != 2:
            print("usage: relabel <clip_id> <subclass>"); return
        try:
            clip_id = int(parts[0])
        except ValueError:
            print("first arg must be a clip id"); return
        self.session.relabel_clip(clip_id, parts[1])
        print(f"relabeled clip {clip_id} as {parts[1]}")

    def do_discard(self, arg: str) -> None:
        """discard <cluster_id>"""
        try:
            cid = int(arg.strip())
        except ValueError:
            print("usage: discard <cluster_id>"); return
        self.session.discard_cluster(cid)
        print(f"discarded cluster {cid}")

    def do_split(self, arg: str) -> None:
        """split <clip_id> — pull a clip into a new singleton cluster."""
        try:
            clip_id = int(arg.strip())
        except ValueError:
            print("usage: split <clip_id>"); return
        new_id = self.session.split_clip_out(clip_id)
        print(f"clip {clip_id} now in singleton cluster {new_id}")

    def do_merge(self, arg: str) -> None:
        """merge <cluster_a> <cluster_b>"""
        parts = shlex.split(arg)
        if len(parts) != 2:
            print("usage: merge <cluster_a> <cluster_b>"); return
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            print("both args must be cluster ids"); return
        new_id = self.session.merge_clusters(a, b)
        print(f"merged into cluster {new_id}")

    def do_status(self, arg: str) -> None:
        """status — uncommitted changes summary"""
        changes = self.session.uncommitted_changes()
        if not changes:
            print("(no uncommitted changes)")
            return
        for c in changes:
            print(c)

    def do_commit(self, arg: str) -> None:
        """commit [--note '...']"""
        note = None
        parts = shlex.split(arg)
        if "--note" in parts:
            i = parts.index("--note")
            note = parts[i + 1] if i + 1 < len(parts) else None
        sid = self.session.commit(note=note)
        print(f"committed; bank snapshot id={sid}")

    def do_rollback(self, arg: str) -> None:
        """rollback — discard uncommitted changes"""
        self.session.rollback()
        print("rolled back to session-open snapshot")

    def do_quit(self, arg: str) -> bool:
        return True

    do_exit = do_quit
    do_EOF = do_quit


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m combat_tracker_recognizer review",
        description="Review unknown action clips and label them.",
    )
    p.add_argument("session_id", help="session id to review")
    p.add_argument("--db", default="./recognizer.db", help="ClipStore SQLite path")
    p.add_argument("--parent", default=None, help="filter by parent class")
    args = p.parse_args(argv)

    cfg = RecognizerConfig()
    cfg.store.db_path = args.db
    rec = SubclassActionRecognizer(cfg)
    session = ReviewSession(
        session_id=args.session_id,
        clipstore=rec.clipstore,
        bank=rec.bank,
        config=cfg.review,
        encoder_version=rec.encoder.version,
    )
    shell = ReviewShell(session, parent_filter=args.parent)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n(interrupt — uncommitted changes will be lost; rerun and 'commit' to persist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
