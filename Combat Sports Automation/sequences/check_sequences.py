import pandas as pd

df = pd.read_csv("yolo_predictions_4.csv")
df = df.sort_values(["class_id", "frame"])

# optional: keep only one row per (frame, class) in case YOLO fires twice
df = df.drop_duplicates(subset=["frame", "class_id"])

segments = []
for cid, g in df.groupby("class_id"):

    # identify breaks between consecutive frames
    run_id = (g["frame"].diff() != 1).cumsum()

    # length (in frames) of each contiguous run
    seg_lens = g.groupby(run_id).size()
    segments.extend(seg_lens)

summary = pd.Series(segments).describe(percentiles=[.5, .9, .95, .99])
print(summary)
