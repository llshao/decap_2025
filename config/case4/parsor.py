#!/usr/bin/env python3
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

HEADER_KEYS = ["Title:", "Date:", "Plotname:", "Flags:", "No. Variables:", "No. Points:"]

def _parse_float_pair(s: str) -> Tuple[float, float]:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p for p in re.split(r"[,\s]+", s) if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse complex pair from '{s}'")
    return float(parts[0]), float(parts[1])

def _read_line(lines: List[str], idx: int) -> Tuple[str, int]:
    if idx >= len(lines):
        raise EOFError("Unexpected end of file while reading .raw")
    return lines[idx].rstrip("\n"), idx + 1

def _skip_blank(lines: List[str], idx: int) -> int:
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return idx

def parse_one_plot(lines: List[str], start_idx: int):
    idx = start_idx
    meta = {}
    while True:
        line, idx = _read_line(lines, idx)
        if not line.strip():
            continue
        if line.startswith("Variables:"):
            break
        for key in HEADER_KEYS:
            if line.startswith(key):
                meta[key[:-1].lower()] = line[len(key):].strip()
                break
    nvars = int(meta.get("no. variables", "0"))
    variables = []
    for _ in range(nvars):
        line, idx = _read_line(lines, idx)
        parts = [p for p in re.split(r"\s+", line.strip()) if p]
        if len(parts) < 3:
            raise ValueError(f"Bad variable line: '{line}'")
        vindex = int(parts[0])
        vname = parts[1]
        vtype = parts[2]
        variables.append((vindex, vname, vtype))

    idx = _skip_blank(lines, idx)
    line, idx = _read_line(lines, idx)
    if not line.startswith("Values:"):
        raise ValueError(f"Expected 'Values:' but got '{line}'")

    npoints = int(meta.get("no. points", "0"))
    flags = meta.get("flags", "").lower()
    is_complex = "complex" in flags

    data = {name: [] for _, name, _ in variables}

    for _pt in range(npoints):
        idx = _skip_blank(lines, idx)
        line, idx = _read_line(lines, idx)
        first_value_inline = None
        m_plain = re.match(r"^\s*(?:Index\s+)?(\d+)\s*$", line)
        m_inline = re.match(r"^\s*(?:Index\s+)?(\d+)\s*\t\s*(.*\S)\s*$", line)
        if m_inline:
            first_value_inline = m_inline.group(2)
        elif not m_plain:
            try:
                float(line.strip())
                idx -= 1
            except Exception:
                raise ValueError(f"Expected point index or value, got '{line}'")
        for vidx, (vindex, name, vtype) in enumerate(variables):
            if vidx == 0 and first_value_inline is not None:
                txt = first_value_inline.strip()
            else:
                idx = _skip_blank(lines, idx)
                line, idx = _read_line(lines, idx)
                txt = line.strip()
            if is_complex:
                try:
                    re_val, im_val = _parse_float_pair(txt)
                except Exception:
                    nxt, idx = _read_line(lines, idx)
                    txt2 = (txt + " " + nxt).strip()
                    re_val, im_val = _parse_float_pair(txt2)
                data[name].append((re_val, im_val))
            else:
                try:
                    val = float(txt.split()[0])
                except Exception:
                    nxt, idx = _read_line(lines, idx)
                    txt2 = (txt + " " + nxt).strip()
                    val = float(txt2.split()[0])
                data[name].append(val)

    plot = {
        "title": meta.get("title", ""),
        "date": meta.get("date", ""),
        "plotname": meta.get("plotname", ""),
        "flags": flags,
        "nvars": nvars,
        "npoints": npoints,
        "variables": variables,
        "data": data,
    }
    return plot, idx

def parse_raw_ascii(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    plots = []
    idx = 0
    while idx < len(lines):
        while idx < len(lines) and not lines[idx].startswith("Title:"):
            idx += 1
        if idx >= len(lines):
            break
        plot, idx = parse_one_plot(lines, idx)
        plots.append(plot)
    if not plots:
        raise ValueError("No plots found in the .raw file. Is it ASCII format?")
    return plots

def plots_summary(plots):
    rows = []
    for i, p in enumerate(plots):
        rows.append({
            "index": i,
            "plotname": p["plotname"],
            "title": p["title"],
            "flags": p["flags"],
            "nvars": p["nvars"],
            "npoints": p["npoints"],
        })
    return pd.DataFrame(rows)

def export_plot_to_csv(plot, out_csv: Path, split_complex=True):
    vars_meta = plot["variables"]
    is_complex = "complex" in plot["flags"]
    columns = []
    data_cols = {}
    for _, name, _ in vars_meta:
        series = plot["data"][name]
        if is_complex and split_complex:
            data_cols[f"{name}_re"] = [v[0] for v in series]
            data_cols[f"{name}_im"] = [v[1] for v in series]
            columns.extend([f"{name}_re", f"{name}_im"])
        else:
            data_cols[name] = series
            columns.append(name)
    df = pd.DataFrame({k: data_cols[k] for k in columns})
    df.to_csv(out_csv, index=False)
    return df

# -------- Visualization helpers --------
def _has_complex(df, base):
    return f"{base}_re" in df.columns and f"{base}_im" in df.columns

def _mag(df, base):
    re_v = df[f"{base}_re"].to_numpy(dtype=float)
    im_v = df[f"{base}_im"].to_numpy(dtype=float)
    return np.sqrt(re_v*re_v + im_v*im_v)

def _phase_deg(df, base):
    re_v = df[f"{base}_re"].to_numpy(dtype=float)
    im_v = df[f"{base}_im"].to_numpy(dtype=float)
    return np.degrees(np.arctan2(im_v, re_v))

def _nearest_index(x, v):
    x = np.asarray(x, dtype=float)
    return int(np.argmin(np.abs(x - float(v))))

def bode_plots(df, out_dir: Path, out_base="bode_vout", node_name="v(out)"):
    if not _has_complex(df, node_name):
        return None, None
    freq = df["frequency"].to_numpy(dtype=float)
    mag = _mag(df, node_name)
    ph = _phase_deg(df, node_name)

    p1 = out_dir / f"{out_base}_mag.png"
    plt.figure()
    plt.loglog(freq, mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(f"|{node_name}|")
    plt.title(f"Bode Magnitude of {node_name}")
    plt.savefig(p1, bbox_inches="tight")
    plt.close()

    p2 = out_dir / f"{out_base}_phase.png"
    plt.figure()
    plt.semilogx(freq, ph)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.title(f"Bode Phase of {node_name}")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    return p1, p2

def plot_names(df, out_dir: Path, names: list, prefix="sens"):
    freq = df["frequency"].to_numpy(dtype=float)
    outs = []
    for nm in names:
        if not _has_complex(df, nm):
            continue
        p = out_dir / f"{prefix}_{nm}_mag.png"
        plt.figure()
        plt.loglog(freq, _mag(df, nm))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(f"|{nm}|")
        plt.title(f"Sensitivity Magnitude: {nm}")
        plt.savefig(p, bbox_inches="tight")
        plt.close()
        outs.append(p)
    return outs

def topk_by_freq(df, out_dir: Path, k: int, at_freq: float, exclude_prefixes=("frequency",), exclude_exact=("v(out)",)):
    freq = df["frequency"].to_numpy(dtype=float)
    idx = _nearest_index(freq, at_freq)
    base_names = []
    for col in df.columns:
        if col.endswith("_re"):
            base = col[:-3]
            if _has_complex(df, base):
                base_names.append(base)
    base_names = sorted(set(base_names))
    filtered = []
    for b in base_names:
        if b in exclude_exact:
            continue
        if any(b.startswith(pfx) for pfx in exclude_prefixes):
            continue
        filtered.append(b)
    vals = []
    for b in filtered:
        mag = _mag(df, b)
        vals.append((b, float(mag[idx])))
    vals.sort(key=lambda x: x[1], reverse=True)
    top = vals[:k]

    out_imgs = []
    for (b, _) in top:
        p = out_dir / f"topk_mag_{b}.png"
        plt.figure()
        plt.loglog(freq, _mag(df, b))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(f"|{b}|")
        plt.title(f"Sensitivity Magnitude (Top-K) at ~{freq[idx]:.3g} Hz: {b}")
        plt.savefig(p, bbox_inches="tight")
        plt.close()
        out_imgs.append(p)

    labels = [b for (b, _) in top]
    heights = [v for (_, v) in top]
    pbar = out_dir / f"topk_snapshot_{int(round(at_freq))}.png"
    plt.figure()
    plt.bar(range(len(labels)), heights)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(f"|vector| at {freq[idx]:.3g} Hz")
    plt.title(f"Top-{k} Sensitivities at ~{freq[idx]:.3g} Hz")
    plt.tight_layout()
    plt.savefig(pbar, bbox_inches="tight")
    plt.close()
    out_imgs.append(pbar)
    return out_imgs, top

def add_viz_cli(ap):
    ap.add_argument("--bode", action="store_true", help="Plot Bode (mag/phase) for v(out) if present")
    ap.add_argument("--viz-names", nargs="+", default=None, help="List of base names to visualize (magnitudes vs frequency)")
    ap.add_argument("--viz-topk", type=int, default=None, help="Plot top-K sensitivity magnitudes ranked at a reference frequency")
    ap.add_argument("--at-freq", type=float, default=None, help="Reference frequency in Hz for --viz-topk")
    return ap

def handle_viz(df, out_dir: Path, args):
    outputs = []
    notes = []

    if args.bode:
        b1, b2 = bode_plots(df, out_dir)
        if b1 and b2:
            outputs.extend([b1, b2])
        else:
            notes.append("No complex v(out) found; skipped --bode.")

    if args.viz_names:
        imgs = plot_names(df, out_dir, args.viz_names)
        outputs.extend(imgs)
        missing = [nm for nm in args.viz_names if not _has_complex(df, nm)]
        if missing:
            notes.append(f"These names were not found or not complex: {', '.join(missing)}")

    if args.viz_topk:
        ref = args.at_freq if args.at_freq is not None else df['frequency'].iloc[len(df)//2]
        imgs, top = topk_by_freq(df, out_dir, args.viz_topk, ref)
        outputs.extend(imgs)
        notes.append("Top-{} at ~{} Hz: {}".format(args.viz_topk, ref, ", ".join(["{}={:.3g}".format(b, v) for b, v in top])))

    return outputs, notes

def main():
    ap = argparse.ArgumentParser(description="Export an ngspice ASCII .raw plot to CSV + optional visualization")
    ap.add_argument("rawfile", type=Path, help="Path to ASCII .raw file (e.g., rc_all_plots.raw)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output CSV path (default: <rawfile>_<plotidx>.csv)")
    ap.add_argument("--plot", type=int, default=None, help="Plot index to export (default: pick the first plot containing 'sens' in plotname, else 0)")
    ap.add_argument("--list", action="store_true", help="List plots and exit")
    ap = add_viz_cli(ap)
    args = ap.parse_args()

    plots = parse_raw_ascii(args.rawfile)
    if args.list:
        df = plots_summary(plots)
        print(df.to_string(index=False))
        return

    plot_idx = args.plot
    if plot_idx is None:
        cand = [i for i, p in enumerate(plots) if "sens" in p["plotname"].lower()]
        plot_idx = cand[0] if cand else 0

    if not (0 <= plot_idx < len(plots)):
        raise IndexError(f"Plot index {plot_idx} out of range (0..{len(plots)-1})")

    plot = plots[plot_idx]
    out_csv = args.out or args.rawfile.with_suffix("")
    if args.out is None:
        out_csv = out_csv.parent / f"{out_csv.name}_plot{plot_idx}.csv"

    df = export_plot_to_csv(plot, out_csv)
    print(f"Exported plot {plot_idx} ('{plot['plotname']}') to: {out_csv}")

    out_dir = out_csv.parent
    viz_imgs, viz_notes = handle_viz(df, out_dir, args)
    for img in viz_imgs:
        print(f"Saved figure: {img}")
    for nt in viz_notes:
        print(f"Note: {nt}")

if __name__ == "__main__":
    main()
