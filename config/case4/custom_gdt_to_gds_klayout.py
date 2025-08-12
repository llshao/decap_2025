# custom_gdt_to_gds_klayout.py
# Purpose: Parse a simplified text layout format (like the provided inv.gdt.rtf content)
#          and export to standard GDSII using KLayout's Python API (pya).
#
# Usage (Option A - recommended):
#   klayout -b -r custom_gdt_to_gds_klayout.py -rd IN=/path/to/inv.gdt.rtf -rd OUT=/path/to/out.gds
#
# Usage (Option B - if 'pya' importable in system python):
#   python3 custom_gdt_to_gds_klayout.py /path/to/inv.gdt.rtf /path/to/out.gds
#
# Notes:
#   - Assumes coordinates in micrometers (µm). We set layout.dbu = 0.001 µm (== 1 nm per DBU).
#   - Supports two primitives found in the example: rectangles/polygons `b{<layer> xy(...)}`
#     and texts `t{<layer> ... xy(x y) 'STRING'}`.
#   - Unknown tokens like 'tt251', 'mc', 'm0.05' are parsed best-effort:
#       * If 'm<value>' exists, it is treated as text height (in µm). Otherwise default 0.2 µm.
#   - All shapes go into datatype/texttype = 0.

import os
import re
import sys

# Try importing pya from KLayout
try:
    import pya
except Exception as e:
    sys.stderr.write("ERROR: Could not import KLayout's 'pya'. Run via 'klayout -b -r ...'.\n")
    raise

def get_cli_paths():
    # 1) KLayout -rd style variables
    in_path = pya.Application.instance().get_config("IN")
    out_path = pya.Application.instance().get_config("OUT")
    # 2) Fallback to argv
    if (not in_path or not out_path) and len(sys.argv) >= 3:
        in_path = in_path or sys.argv[1]
        out_path = out_path or sys.argv[2]
    if not in_path or not out_path:
        raise SystemExit(
            "Usage:\n"
            "  klayout -b -r custom_gdt_to_gds_klayout.py -rd IN=INPUT -rd OUT=OUTPUT.gds\n"
            "or\n"
            "  python3 custom_gdt_to_gds_klayout.py INPUT OUTPUT.gds\n"
        )
    return in_path, out_path

def parse_text_file(text):
    """
    Extract:
      - cell name from: cell{ ... 'CELLNAME' }
      - polygons/rects from: b{<layer> xy(x1 y1 x2 y2 ... xN yN)}
      - texts from: t{<layer> ... xy(x y) 'STRING'}
    Returns dict: { 'cell': name, 'polys': [(layer, [(x,y),...])], 'texts': [(layer, x, y, s, height_um)] }
    """
    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", text)

    # Cell name
    cell_name = "TOP"
    m = re.search(r"cell\s*\{[^}]*'([^']+)'", t, re.IGNORECASE | re.DOTALL)
    if m:
        cell_name = m.group(1)

    polys = []
    texts = []

    # b{<layer> xy(x y ...)}  e.g., b{12 xy(0.08 0.70 0.72 0.70 0.72 1.05 0.08 1.05)}
    for bm in re.finditer(r"b\\?\{(\d+)\s+xy\(\s*([^)]+?)\s*\)\s*\\?\}", t, re.IGNORECASE):
        layer = int(bm.group(1))
        coords_str = bm.group(2).strip()
        nums = [float(v) for v in re.split(r"[\s,]+", coords_str) if v]
        if len(nums) % 2 != 0:
            continue  # skip malformed
        pts = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
        # Close polygon if not closed
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        polys.append((layer, pts))

    # t{<layer> ... xy(x y) 'STRING'}
    # Also try to read 'm<height>' token if present, e.g. m0.05 -> 0.05 µm
    t_re = re.compile(
        r"t\\?\{(\d+)\s+(.*?)xy\(\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*\)\s*'([^']+)'\s*\\?\}",
        re.IGNORECASE
    )
    for tm in t_re.finditer(t):
        layer = int(tm.group(1))
        token_blob = tm.group(2) or ""
        x = float(tm.group(3))
        y = float(tm.group(4))
        s = tm.group(5)
        # text height
        height_um = 0.2
        mh = re.search(r"\bm([0-9]*\.?[0-9]+)\b", token_blob)
        if mh:
            try:
                height_um = float(mh.group(1))
            except:
                pass
        texts.append((layer, x, y, s, height_um))

    return {"cell": cell_name, "polys": polys, "texts": texts}

def um_to_dbu(v_um, dbu_um):
    # Convert micrometers to integer DBU
    return int(round(v_um / dbu_um))

def build_layout(parsed, out_gds):
    # Setup layout
    ly = pya.Layout()
    dbu_um = 0.001  # 1 DBU = 0.001 µm = 1 nm
    ly.dbu = dbu_um
    top = ly.create_cell(parsed["cell"] or "TOP")

    # Helper to get layer index
    layer_cache = {}
    def layer_index(l):
        if l not in layer_cache:
            layer_cache[l] = ly.layer(l, 0)
        return layer_cache[l]

    # Insert polygons
    for layer, pts in parsed["polys"]:
        li = layer_index(layer)
        pts_dbu = [pya.Point(um_to_dbu(x, dbu_um), um_to_dbu(y, dbu_um)) for x, y in pts]
        # Use polygon
        polygon = pya.Polygon(pts_dbu)
        top.shapes(li).insert(polygon)

    # Insert texts
    for layer, x, y, s, h_um in parsed["texts"]:
        li = layer_index(layer)
        # KLayout's Text() uses kdb units; use DBU coords
        tx = um_to_dbu(x, dbu_um)
        ty = um_to_dbu(y, dbu_um)
        text_obj = pya.Text(s, pya.Trans(pya.Point(tx, ty)))
        # Size control: use TextGenerator to create text as polygons with given height if needed.
        # For now, insert as TEXT label, then set size via properties if desired.
        top.shapes(li).insert(text_obj)
        # Optional: If you prefer drawn polygons for guaranteed GDS compatibility:
        # tg = pya.TextGenerator.default_generator()
        # text_poly = tg.text(s, int(round(h_um / dbu_um)), False, 0, 0)
        # top.shapes(li).insert(pya.DPolygon(text_poly.transformed(pya.Trans(tx, ty))))

    # Save to GDS
    opt = pya.SaveLayoutOptions()
    opt.format = "GDS2"
    ly.write(out_gds, opt)
    return out_gds

def main():
    in_path = "/Users/leilaishao/Downloads/inv.rtf"
    out_path = "/Users/leilaishao/Downloads/out.gds"
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    parsed = parse_text_file(content)
    out_gds = build_layout(parsed, out_path)
    print(f"[OK] Wrote GDS: {out_gds}")
    print(f"Cell: {parsed['cell']} | polys: {len(parsed['polys'])} | texts: {len(parsed['texts'])}")

if __name__ == "__main__":
    main()

# Enter your Python code here

