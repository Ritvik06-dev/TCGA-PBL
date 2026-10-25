"""Download one TCGA slide and cut out the exact 256x256 patches whose UNI2-h features we used.
Run from the tcga folder:  python get_patches.py
"""
import glob, json, os, shutil, urllib.parse, urllib.request
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openslide

N_PATCHES, PATCH, MAG = 12, 256, 20
GDC = "https://api.gdc.cancer.gov"
os.makedirs("slides", exist_ok=True)
os.makedirs("patches", exist_ok=True)

# ---------- 1. candidate feature files from our cohort
pids = set(pd.read_csv("cohort_final.csv")["bcr_patient_barcode"])
h5s = [f for f in sorted(glob.glob(os.path.join("features", "*.h5"))) if os.path.basename(f)[:12] in pids][:25]
names = {os.path.basename(f)[:-3]: f for f in h5s}

# ---------- 2. ask GDC for the matching slides and pick the smallest one
filt = {"op": "in", "content": {"field": "file_name",
        "value": [n + ".svs" for n in names] + list(names)}}
q = urllib.parse.urlencode({"filters": json.dumps(filt), "fields": "file_id,file_name,file_size",
                            "format": "json", "size": "100"})
hits = json.load(urllib.request.urlopen(f"{GDC}/files?{q}"))["data"]["hits"]
if not hits:
    raise SystemExit("No matching slides found on GDC - send me one feature file name from the features folder.")
hit = min(hits, key=lambda h: h["file_size"])
slide_name = hit["file_name"]
h5_path = names.get(slide_name[:-4], names.get(slide_name))
print(f"Slide: {slide_name} ({hit['file_size'] / 1e6:.0f} MB)\nFeatures: {h5_path}")

# ---------- 3. download the slide (skipped if already downloaded)
svs = os.path.join("slides", slide_name)
if not os.path.exists(svs):
    print("Downloading slide ...")
    with urllib.request.urlopen(f"{GDC}/data/{hit['file_id']}") as r, open(svs, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)

# ---------- 4. read patch positions and cut the patches
slide = openslide.OpenSlide(svs)
w0, h0 = slide.dimensions
mag = float(slide.properties.get("openslide.objective-power", 40))
size0 = int(round(PATCH * mag / MAG))            # 256 px at 20x = 512 px on a 40x slide
with h5py.File(h5_path, "r") as h:
    coords = np.asarray(h["coords"]).squeeze().astype(int)
    if (coords[:, 0].max() > w0 or coords[:, 1].max() > h0) and "coords_patching" in h:
        coords = np.asarray(h["coords_patching"]).squeeze().astype(int)
print(f"{len(coords)} patches in this slide; slide is {w0} x {h0} px at {mag:.0f}x")

rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(coords), size=min(N_PATCHES, len(coords)), replace=False))
tiles = []
for i in idx:
    x, y = coords[i]
    img = slide.read_region((int(x), int(y)), 0, (size0, size0)).convert("RGB").resize((PATCH, PATCH))
    img.save(os.path.join("patches", f"patch_{i}.png"))
    tiles.append((i, x, y, img))

# ---------- 5. save a grid for the presentation
cols = 4
rows = int(np.ceil(len(tiles) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
for ax in axes.ravel():
    ax.axis("off")
for ax, (i, x, y, img) in zip(axes.ravel(), tiles):
    ax.imshow(img)
    ax.set_title(f"patch {i}\n(x={x}, y={y})", fontsize=8)
fig.suptitle(f"{slide_name[:23]}: {len(tiles)} of {len(coords)} patches (256x256 at 20x)", fontsize=10)
fig.tight_layout()
fig.savefig("patch_grid.png", dpi=200)
thumb = slide.get_thumbnail((1200, 1200))
thumb.save("slide_thumbnail.png")
print("Saved patch_grid.png, slide_thumbnail.png and patches/*.png")