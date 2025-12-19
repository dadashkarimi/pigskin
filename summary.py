# summarize_two_datasets.py
import os, re, sys
import nibabel as nib
import pandas as pd

# ---- CONFIG ----
# roots to scan (override via CLI if you want)
DEFAULT_ROOTS = ["results", "results_doug"]

# acceptable image basenames (first found wins)
IMAGE_CANDIDATES = ["image.nii.gz", "image.nii"]

# which tags count as healthy vs post-injury
HEALTHY_TAGS = {"pre"}
POST_TAGS = {"post", "1month", "3day", "3month", "6month"}

# ----------------

def parse_folder_cci(name):
    """
    CCI naming: JAW-076_pre, JAW-106_6month, etc.
    Returns (subject, tag) or (None, None) if not matched.
    """
    m = re.match(r"^(JAW-\d+)_([A-Za-z0-9]+)$", name)
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()

def parse_folder_rotational(name):
    """
    Rotational naming: 2016-12_pre, 2016-12_post, 2017-1_pre, etc.
    Subject id = the first part (e.g., 2016-12), tag = pre/post
    """
    m = re.match(r"^(\d{4}-\d+)_([A-Za-z0-9]+)$", name)
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()

def find_image(folder):
    for fn in IMAGE_CANDIDATES:
        p = os.path.join(folder, fn)
        if os.path.isfile(p):
            return p
    return None

def summarize_root(root, dataset_label):
    rows = []
    subjects = set()
    per_tag = {}
    healthy_sessions = 0
    post_sessions = 0

    # choose parser based on dataset label (but we’ll try both just in case)
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue

        subj = tag = None
        if dataset_label.lower().startswith("cci"):
            subj, tag = parse_folder_cci(d)
            if subj is None:  # fallback try rotational pattern
                subj, tag = parse_folder_rotational(d)
        else:
            subj, tag = parse_folder_rotational(d)
            if subj is None:  # fallback try CCI pattern
                subj, tag = parse_folder_cci(d)

        if subj is None:
            # skip non-matching folders
            continue

        subjects.add(subj)
        img_path = find_image(full)

        rec = {
            "dataset": dataset_label,
            "subject": subj,
            "session_tag": tag,
            "path": full,
            "image_path": img_path if img_path else "",
            "dim_x": "", "dim_y": "", "dim_z": "",
            "vox_x": "", "vox_y": "", "vox_z": "",
            "status": "MISSING image" if img_path is None else "OK",
        }

        if img_path:
            img = nib.load(img_path)
            shape = img.shape
            zooms = img.header.get_zooms()[:min(3, len(shape))]
            if len(shape) >= 3:
                rec["dim_x"], rec["dim_y"], rec["dim_z"] = shape[:3]
                # zooms may be length<3 if header odd—pad safely
                z = list(zooms) + [None, None, None]
                rec["vox_x"], rec["vox_y"], rec["vox_z"] = z[:3]

        # counts
        per_tag[tag] = per_tag.get(tag, 0) + 1
        if tag in HEALTHY_TAGS:
            healthy_sessions += 1
        if tag in POST_TAGS:
            post_sessions += 1

        rows.append(rec)

    summary = {
        "dataset": dataset_label,
        "unique_subjects": len(subjects),
        "total_sessions": len(rows),
        "healthy_sessions": healthy_sessions,
        "post_sessions": post_sessions,
        "by_tag": per_tag,
    }
    return rows, summary

def main():
    roots = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ROOTS

    all_rows = []
    summaries = []

    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] Skipping missing root: {root}")
            continue
        # infer dataset label from folder name
        label = "Rotational" if "doug" in os.path.basename(root).lower() else "CCI"
        rows, summ = summarize_root(root, label)
        all_rows.extend(rows)
        summaries.append(summ)

    if not all_rows:
        print("No data found. Check roots / folder names.")
        return

    df = pd.DataFrame(all_rows).sort_values(["dataset", "subject", "session_tag"])
    out_csv = "combined_results_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSummary written to: {out_csv}\n")

    # print human-readable summaries
    for summ in summaries:
        print(f"=== {summ['dataset']} ===")
        print(f"Unique subjects: {summ['unique_subjects']}")
        print(f"Total sessions: {summ['total_sessions']}")
        print(f"Healthy sessions (pre): {summ['healthy_sessions']}")
        print(f"Post-injury sessions: {summ['post_sessions']}")
        print("By tag:")
        for k in sorted(summ["by_tag"].keys()):
            print(f"  {k}: {summ['by_tag'][k]}")
        print()

if __name__ == "__main__":
    main()

