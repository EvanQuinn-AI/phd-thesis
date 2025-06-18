# check_sequences.py
import sys, pandas as pd, numpy as np, pathlib as pl

# ---------- configure ----------
# class-ids that correspond to *actions*, not just presence
ACTION_CLASSES = {1: "high-guard", 2: "kick-knee", 5: "punch"}
PCTS          = [.5, .9, .95, .99]

# ---------- helpers ----------
def contiguous_lengths(frames: np.ndarray) -> np.ndarray:
    """Return lengths of contiguous frame runs."""
    if len(frames) == 0:
        return np.array([], dtype=int)
    breaks = np.where(np.diff(frames) != 1)[0] + 1
    idx    = np.concatenate(([0], breaks, [len(frames)]))
    return np.diff(idx)

def describe_lengths(lengths: list[int]) -> pd.Series:
    s = pd.Series(lengths, dtype=float)
    return s.describe(percentiles=PCTS).round(1)

# ---------- driver ----------
def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    # keep only rows that belong to action classes we care about
    df = df[df["class_id"].isin(ACTION_CLASSES)]

    seg_lens_global = []
    print(f"\nFile: {pl.Path(csv_path).name}")
    for cid, name in ACTION_CLASSES.items():
        frames = df.loc[df["class_id"] == cid, "frame"].drop_duplicates().to_numpy()
        lengths = contiguous_lengths(np.sort(frames))
        seg_lens_global.extend(lengths)

        if len(lengths):
            print(f"\n{cid}  {name}")
            print(describe_lengths(lengths))

    print("\n--- all actions combined ---")
    print(describe_lengths(seg_lens_global))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python check_sequences.py <yolo_csv>")
    main(sys.argv[1])
