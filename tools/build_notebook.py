"""Builds notebooks/dhaka_traffic_density.ipynb (sectioned). Run: python tools/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# 🚦 Dhaka Traffic Density Analyzer: YOLO11 + ByteTrack + congestion score

Detection alone is not the goal. This notebook **counts vehicles**, turns the counts into a **congestion score (0–100)** and a
**heatmap**, and counts unique vehicles in video with **ByteTrack**. The model knows local classes such as **rickshaw**,
**CNG**, **human hauler (leguna)** and **auto rickshaw**.

| Step | What happens |
|---|---|
| Data | **Dhaka-AI** (DatasetNinja, 3,953 images, 21 classes). 956 unlabeled images are **filtered out** |
| Clean | duplicate images removed, EXIF rotation fixed, boxes clipped, own train/val/test split (80/10/10) |
| Train | YOLO11 fine-tune on 2×T4, **checkpoints mirrored to your GCS bucket → resumes after a stop** |
| Analyze | per-class AP · congestion score + heatmap on images · ByteTrack counting on video |
| Ship | `.pt` / `.pth` / ONNX weights + Gradio demo (`app/`) |

**Kaggle settings:** GPU **T4 ×2** · Internet **On** · Add-ons → Google Cloud Services → Cloud Storage.
Runtime ≈ 2.5–3.5 h for 80 epochs (if a session stops, *Run All* again to resume).
""")

md("## 1 · Setup")
code("""
!pip install -q ultralytics lap gradio onnx onnxslim onnxruntime tabulate 2>&1 | tail -1
import glob, hashlib, json, os, re, shutil, sys, tarfile, threading, time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from IPython.display import display
from ultralytics import YOLO, settings

for k in ("wandb", "mlflow", "comet", "clearml", "neptune", "dvc"):      # no external loggers
    try:
        settings.update({k: False})
    except Exception:
        pass
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(N_GPU)])
""")

md("## 2 · Config\nChange only this cell.")
code("""
CFG = dict(
    project_id="mimic-cxr-training",          # your GCP project (bucket for checkpoints)
    bucket="mimic-cxr-training-mimic-ckpt",   # reused private bucket; prefix keeps projects apart
    prefix="dhaka-traffic",
    use_gcs=True,
    model="yolo11s.pt",                       # yolo11n/s/m; s = good accuracy/speed on T4
    imgsz=960,                                # many small vehicles -> larger than the default 640
    epochs=80, patience=25,
    batch=16 * max(N_GPU, 1),                 # 16 per GPU
    cache="ram",                              # faster epochs; set False if RAM is short
    seed=42,
    video="",                                 # path to a traffic video (else auto-find in /kaggle/input)
    video_seconds=20,
    demo=True,                                # launch the Gradio demo at the end
)
OUT, RUNS, DATA = Path("outputs").resolve(), Path("runs").resolve(), Path("/tmp/dhaka_yolo")
for d in ("models", "results/figures", "app"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
Path("app").mkdir(exist_ok=True)
NAME = f"{Path(CFG['model']).stem}-{CFG['imgsz']}"
print("run:", NAME, "| outputs ->", OUT)
""")

md("## 3 · Checkpoint store\nGCS bucket when the Kaggle add-on is attached, otherwise a local folder. "
   "A background thread copies `last.pt` / `best.pt` to the bucket every 2 minutes during training.")
code("""
def gcs_credentials():
    if "KAGGLE_USER_SECRETS_TOKEN" not in os.environ:
        return None
    from kaggle_secrets import UserSecretsClient
    if "GCS" in os.environ.get("KAGGLE_KERNEL_INTEGRATIONS", "").upper().split(":"):
        from kaggle_gcp import KaggleKernelCredentials
        from kaggle_secrets import GcpTarget
        return KaggleKernelCredentials(target=GcpTarget.GCS)
    from google.oauth2.credentials import Credentials            # "Google Cloud SDK" add-on
    return Credentials.from_authorized_user_info(json.loads(UserSecretsClient().get_gcloud_credential()))


class Store:
    def __init__(self, cfg):
        self.prefix, self.bucket, self.local = cfg["prefix"], None, OUT / "store"
        if not cfg["use_gcs"]:
            return
        try:
            from google.api_core import exceptions as gexc
            from google.cloud import storage
            client = storage.Client(project=cfg["project_id"], credentials=gcs_credentials())
            b = client.bucket(cfg["bucket"])
            try:
                exists = b.exists()
            except gexc.Forbidden:
                exists = True
            if not exists:
                b.iam_configuration.uniform_bucket_level_access_enabled = True
                b.iam_configuration.public_access_prevention = "enforced"
                b = client.create_bucket(b, location="US")
            self.bucket = b
            print(f"✅ checkpoints -> gs://{cfg['bucket']}/{self.prefix}")
        except Exception as e:
            print(f"⚠️ GCS unavailable ({type(e).__name__}: {str(e)[:150]}) -> local {self.local}")

    def put(self, local, key):
        if self.bucket:
            self.bucket.blob(f"{self.prefix}/{key}").upload_from_filename(str(local), timeout=600)
        else:
            (self.local / key).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, self.local / key)

    def get(self, key, local) -> bool:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        if self.bucket:
            bl = self.bucket.blob(f"{self.prefix}/{key}")
            if not bl.exists():
                return False
            bl.download_to_filename(str(local), timeout=600)
            return True
        src = self.local / key
        if src.exists() and src.resolve() != Path(local).resolve():
            shutil.copy2(src, local)
        return src.exists()


class Syncer(threading.Thread):
    \"\"\"Mirrors the training run to the store while YOLO trains (works with multi-GPU DDP too).\"\"\"
    FILES = ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml")

    def __init__(self, run_dir, key, every=120):
        super().__init__(daemon=True)
        self.run_dir, self.key, self.every, self.seen, self.halt = Path(run_dir), key, every, {}, threading.Event()

    def sync(self):
        for f in self.FILES:
            p = self.run_dir / f
            if not p.exists() or self.seen.get(f) == p.stat().st_mtime:
                continue
            m = p.stat().st_mtime
            time.sleep(3)
            if p.stat().st_mtime != m:
                continue                                         # still being written -> next round
            tmp = Path("/tmp/sync") / f
            tmp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, tmp)
            store.put(tmp, f"{self.key}/{Path(f).name}")
            self.seen[f] = m

    def run(self):
        while not self.halt.wait(self.every):
            try:
                self.sync()
            except Exception as e:
                print("sync warning:", type(e).__name__, str(e)[:120])

    def finish(self):
        self.halt.set()
        self.sync()


store = Store(CFG)
""")

md("## 4 · Get Dhaka-AI\nOrder: cached YOLO copy in your bucket → DatasetNinja archive (Supervisely format) → original Kaggle "
   "version (Pascal VOC). After the first run the cleaned dataset comes straight from the bucket.")
code("""
CLASSES = ["car", "bus", "motorbike", "three wheelers (CNG)", "rickshaw", "truck", "pickup", "minivan", "suv", "van",
           "bicycle", "auto rickshaw", "human hauler", "wheelbarrow", "ambulance", "minibus", "taxi", "army vehicle",
           "scooter", "policecar", "garbagevan"]
KEY = {re.sub(r"[^a-z]", "", c.lower()): i for i, c in enumerate(CLASSES)}
NINJA_URL = ("https://www.dropbox.com/scl/fi/3apbbrlsd2fkkvq6xi088/dhaka-ai-DatasetNinja.tar"
             "?rlkey=4sa6qehntce2fdhpe3bvd4d3r&dl=1")
KAGGLE_ORIGINAL = "rifat963/dhakaai-dhaka-based-traffic-detection-dataset"
DATA_TAR = "data/dhaka_yolo_v1.tar"


def download(url, dst):
    import requests
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, done, t0, mark = int(r.headers.get("content-length", 0)), 0, time.time(), 0
        with open(dst, "wb") as f:
            for chunk in r.iter_content(8 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= mark:
                    print(f"  {done / 1e9:.2f}/{total / 1e9:.2f} GB | {done / 1e6 / (time.time() - t0):.0f} MB/s")
                    mark += 0.2


RAW_STATS = Counter()


def parse_voc(x: Path):
    \"\"\"Pascal VOC XML -> ElementTree root, or None for empty / non-XML / broken files.\"\"\"
    b = x.read_bytes().lstrip(b"\\xef\\xbb\\xbf \\t\\r\\n\\x00")
    if not b.startswith(b"<"):
        return None
    try:
        return ET.fromstring(b)
    except ET.ParseError:
        try:
            from lxml import etree as LET                        # tolerant parser for slightly broken files
            r = LET.fromstring(b, LET.XMLParser(recover=True))
            return None if r is None else ET.fromstring(LET.tostring(r))
        except Exception:
            return None


def raw_items():
    \"\"\"Yields (image_path, [(class_name, x1, y1, x2, y2), ...]) from whichever source is available.\"\"\"
    raw = Path("/tmp/dhaka_raw")
    try:
        if not any(raw.rglob("*.json")):
            raw.mkdir(parents=True, exist_ok=True)
            print("Downloading DatasetNinja archive (~2 GB)...")
            download(NINJA_URL, "/tmp/dhaka.tar")
            with tarfile.open("/tmp/dhaka.tar") as t:
                t.extractall(raw, filter="data")
            os.remove("/tmp/dhaka.tar")
        anns = sorted(raw.rglob("ann/*.json"))
        if not anns:
            raise RuntimeError("archive has no ann/*.json files")
        print(f"Source: DatasetNinja (Supervisely) | {len(anns)} annotation files")
        for a in anns:
            d = json.loads(a.read_text())
            boxes = [(o["classTitle"], *o["points"]["exterior"][0], *o["points"]["exterior"][1])
                     for o in d.get("objects", []) if o.get("geometryType") == "rectangle"]
            yield a.parent.parent / "img" / a.name[:-5], boxes
        return
    except Exception as e:
        print(f"DatasetNinja failed ({type(e).__name__}: {str(e)[:150]}) -> original Kaggle version")
    import kagglehub
    root = Path(kagglehub.dataset_download(KAGGLE_ORIGINAL))
    imgs = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            imgs.setdefault(p.name.lower(), []).append(p)
    xmls = sorted(root.rglob("*.xml"))
    print(f"Source: Kaggle original (Pascal VOC) | {len(xmls)} XML files, {sum(map(len, imgs.values()))} images")

    def find_image(x, fname):
        names = [fname.lower()] if fname else []
        names += [x.stem.lower() + e for e in (".jpg", ".jpeg", ".png")]
        for n in names:
            cands = imgs.get(n, [])
            if cands:                                            # prefer the image closest to the XML
                return min(cands, key=lambda c: len(set(c.parts) ^ set(x.parts)))
        return None

    for x in xmls:
        t = parse_voc(x)
        if t is None:
            RAW_STATS["unreadable XML (skipped)"] += 1
            continue
        img = find_image(x, (t.findtext("filename") or "").strip())
        if img is None:
            RAW_STATS["XML without image (skipped)"] += 1
            continue
        boxes = []
        for o in t.iter("object"):
            bb = o.find("bndbox")
            try:
                boxes.append((o.findtext("name"), *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
            except (AttributeError, TypeError, ValueError):
                RAW_STATS["broken box (skipped)"] += 1
        yield img, boxes
""")

md("## 5 · Clean + convert to YOLO\nDrops unlabeled images, exact duplicates and unknown class names; fixes EXIF rotation; clips "
   "boxes; splits 80/10/10 by image hash (so a duplicate can never land in two splits).")
code("""
def build_yolo():
    shutil.rmtree(DATA, ignore_errors=True)                      # no leftovers from a crashed run
    stats, seen = Counter(), set()
    for s in ("train", "val", "test"):
        (DATA / "images" / s).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / s).mkdir(parents=True, exist_ok=True)
    for i, (img, boxes) in enumerate(raw_items()):
        stats["images"] += 1
        if not boxes:
            stats["unlabeled (dropped)"] += 1
            continue
        if not Path(img).exists():
            stats["missing file"] += 1
            continue
        h = hashlib.md5(Path(img).read_bytes()).hexdigest()
        if h in seen:
            stats["duplicate (dropped)"] += 1
            continue
        seen.add(h)
        try:
            im = Image.open(img)
            W, H = im.size
        except Exception:
            stats["unreadable"] += 1
            continue
        lines = []
        for name, x1, y1, x2, y2 in boxes:
            c = KEY.get(re.sub(r"[^a-z]", "", str(name).lower()))
            if c is None:
                stats[f"unknown class '{name}'"] += 1
                continue
            x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
            y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                stats["tiny box (dropped)"] += 1
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        if not lines:
            stats["no valid box (dropped)"] += 1
            continue
        b = int(h, 16) % 100
        split = "train" if b < 80 else "val" if b < 90 else "test"
        stem = f"{i:05d}_{re.sub(r'[^A-Za-z0-9_-]', '_', Path(img).stem)[:40]}"
        dst = DATA / "images" / split / f"{stem}.jpg"
        if im.getexif().get(274, 1) != 1 or im.format != "JPEG":   # boxes are in raw pixels -> drop EXIF rotation
            im.convert("RGB").save(dst, quality=95)
            stats["re-encoded (EXIF/format)"] += 1
        else:
            shutil.copy2(img, dst)
        (DATA / "labels" / split / f"{stem}.txt").write_text("\\n".join(lines) + "\\n")
        stats[f"kept {split}"] += 1
    return stats


if not (DATA / "data.yaml").exists():
    if store.get(DATA_TAR, "/tmp/dhaka_yolo.tar"):
        print("Using cleaned dataset from the bucket")
        with tarfile.open("/tmp/dhaka_yolo.tar") as t:
            t.extractall(DATA.parent, filter="data")
    else:
        st = build_yolo() + RAW_STATS
        print(pd.Series(st).sort_index().to_string())
        (DATA / "data.yaml").write_text(
            f"path: {DATA}\\ntrain: images/train\\nval: images/val\\ntest: images/test\\nnames:\\n"
            + "".join(f"  {i}: {c}\\n" for i, c in enumerate(CLASSES)))
        with tarfile.open("/tmp/dhaka_yolo.tar", "w") as t:
            t.add(DATA, arcname=DATA.name)
        store.put("/tmp/dhaka_yolo.tar", DATA_TAR)
    Path("/tmp/dhaka_yolo.tar").unlink(missing_ok=True)

inst = pd.DataFrame({s: Counter(int(l.split()[0]) for f in (DATA / "labels" / s).glob("*.txt")
                                for l in f.read_text().split("\\n") if l) for s in ("train", "val", "test")}).fillna(0)
inst = inst.astype(int).rename(index=dict(enumerate(CLASSES))).sort_values("train", ascending=False)
print({s: len(list((DATA / "images" / s).glob("*"))) for s in ("train", "val", "test")}, "images")
inst
""")
md("> Rare classes (ambulance, police car, army vehicle, garbage van) have very few boxes. Expect low AP there. "
   "Report it rather than hide it.")

md("## 6 · Look at the labels")
code("""
def draw_gt(p):
    img = cv2.imread(str(p))[..., ::-1].copy()
    H, W = img.shape[:2]
    for l in (DATA / "labels" / p.parent.name / f"{p.stem}.txt").read_text().split("\\n"):
        if l:
            c, x, y, w, h = map(float, l.split())
            x1, y1, x2, y2 = int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(img, CLASSES[int(c)], (x1, max(12, y1 - 4)), 0, 0.6, (255, 200, 0), 2)
    return img


samples = sorted((DATA / "images/train").glob("*.jpg"))[:6]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for a, p in zip(axes.flat, samples):
    a.imshow(draw_gt(p)); a.set_title(p.name, fontsize=8); a.axis("off")
plt.tight_layout(); plt.show()
""")

md("## 7 · Train YOLO11 (resumable)\n- First run: starts from COCO-pretrained weights.\n"
   "- After a stop: `last.pt` comes back from the bucket and training **continues from the same epoch**.\n"
   "- Finished: a `DONE` marker is written, so later runs skip straight to evaluation.")
code("""
RUN_DIR, CK = RUNS / NAME, f"ckpt/{NAME}"
W = RUN_DIR / "weights"
loaded_epochs = 0
if store.get(f"{CK}/DONE.json", RUN_DIR / "DONE.json"):
    store.get(f"{CK}/best.pt", W / "best.pt"); store.get(f"{CK}/results.csv", RUN_DIR / "results.csv")
    loaded_epochs = len(pd.read_csv(RUN_DIR / "results.csv"))
    print(f"✅ loaded from checkpoint: training already finished ({loaded_epochs} epochs) -> nothing to retrain")
else:
    resume = store.get(f"{CK}/last.pt", W / "last.pt")
    for f in ("results.csv", "args.yaml", "best.pt"):
        store.get(f"{CK}/{f}", (W if f.endswith(".pt") else RUN_DIR) / f)
    if resume and (RUN_DIR / "results.csv").exists():
        loaded_epochs = len(pd.read_csv(RUN_DIR / "results.csv"))
        print(f"↻ loaded checkpoint: {loaded_epochs}/{CFG['epochs']} epochs already done -> continuing")
    syncer = Syncer(RUN_DIR, CK)
    syncer.start()
    t0 = time.time()
    try:
        if resume:
            print("↻ resuming from bucket checkpoint")
            YOLO(str(W / "last.pt")).train(resume=True)
        else:
            YOLO(CFG["model"]).train(
                data=str(DATA / "data.yaml"), epochs=CFG["epochs"], imgsz=CFG["imgsz"], batch=CFG["batch"],
                device=list(range(N_GPU)) if N_GPU > 1 else 0 if N_GPU else "cpu", project=str(RUNS), name=NAME,
                exist_ok=True, patience=CFG["patience"], cos_lr=True, close_mosaic=10, cache=CFG["cache"], workers=4,
                seed=CFG["seed"], plots=True)
    except AssertionError as e:
        if "nothing to resume" not in str(e).lower() and "finished" not in str(e).lower():
            raise
        print("training had already finished:", e)
    finally:
        syncer.finish()
    done = {"epochs_csv_rows": len(pd.read_csv(RUN_DIR / "results.csv")), "minutes_this_session": round((time.time() - t0) / 60)}
    (RUN_DIR / "DONE.json").write_text(json.dumps(done))
    store.put(RUN_DIR / "DONE.json", f"{CK}/DONE.json")
    print("✅ training finished", done)
""")

md("## 8 · Training history + curves\nEvery epoch, also the ones loaded from the checkpoint after a rerun.")
code("""
res = pd.read_csv(RUN_DIR / "results.csv")
res.columns = [c.strip() for c in res.columns]
cols = [c for c in ("epoch", "train/box_loss", "train/cls_loss", "val/box_loss", "metrics/precision(B)",
                    "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)") if c in res]
table = res[cols].rename(columns=lambda c: c.replace("metrics/", "").replace("(B)", "")).round(4)
table["source"] = np.where(table.epoch <= loaded_epochs, "checkpoint", "this session")
table["best"] = np.where(res.index == res["metrics/mAP50-95(B)"].idxmax(), "★", "")
print(f"{len(table)} epochs ({(table.source == 'checkpoint').sum()} loaded from the checkpoint) | ★ = best mAP50-95")
display(table)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for c in [c for c in res.columns if c.startswith("train/")]:
    ax[0].plot(res.epoch, res[c], label=c.split("/")[1])
for c in [c for c in res.columns if c.startswith("val/")]:
    ax[1].plot(res.epoch, res[c], label=c.split("/")[1])
for c in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
    ax[2].plot(res.epoch, res[c], label=c.split("/")[1])
for a, t in zip(ax, ("train loss", "val loss", "val mAP")):
    a.set(title=t, xlabel="epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "results/figures/training_curves.png", dpi=120); plt.show()
""")

md("## 9 · Evaluate on the held-out test split")
code("""
best = YOLO(str(W / "best.pt"))
m = best.val(data=str(DATA / "data.yaml"), split="test", imgsz=CFG["imgsz"], batch=16, device=0 if N_GPU else "cpu",
             project=str(RUNS), name=f"{NAME}-test", exist_ok=True, plots=True, verbose=False)
overall = {"mAP50": m.box.map50, "mAP50-95": m.box.map, "precision": m.box.mp, "recall": m.box.mr,
           "ms_per_image": m.speed["inference"]}
per_class = pd.DataFrame({"class": [CLASSES[i] for i in m.box.ap_class_index], "AP50": m.box.ap50,
                          "AP50-95": m.box.ap}).set_index("class")
per_class = per_class.join(inst[["train", "test"]].rename(columns=lambda c: f"boxes_{c}")).sort_values("AP50", ascending=False)
print(json.dumps({k: round(float(v), 4) for k, v in overall.items()}, indent=1))
per_class
""")
code("""
for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"):
    p = Path(m.save_dir) / f
    if p.exists():
        shutil.copy2(p, OUT / "results/figures" / f)
        plt.figure(figsize=(10, 8)); plt.imshow(plt.imread(p)); plt.axis("off"); plt.show()
""")

md("## 10 · Analyzer: counting, congestion score, heatmap\nThe same file powers the Gradio app. "
   "**Score** = occupancy (vehicle area / frame) + traffic load in PCU (bus = 3, car = 1, rickshaw = 1, motorbike = 0.5…), "
   "plus slowness for video. It is a transparent heuristic you can tune per camera, not a learned label.")
code("%%writefile app/traffic.py\n" + (ROOT / "app/traffic.py").read_text(encoding="utf-8").rstrip())
code("""
for _m in ("traffic", "app"):
    sys.modules.pop(_m, None)                                    # always use the freshly written file
sys.path.insert(0, str(Path("app").resolve()))
from traffic import Analyzer

an = Analyzer(str(W / "best.pt"), imgsz=CFG["imgsz"], device=0 if N_GPU else "cpu")
test_imgs = sorted((DATA / "images/test").glob("*.jpg"), key=lambda p: -len((DATA / "labels/test" / f"{p.stem}.txt").read_text()))
rows = []
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
for k, p in enumerate(test_imgs[:4]):
    ann, heat, s = an.image(cv2.imread(str(p)))
    rows.append({"image": p.name, **{k2: v for k2, v in s.items() if k2 != "counts"}, "top classes": dict(list(s["counts"].items())[:4])})
    axes[k, 0].imshow(ann[..., ::-1]); axes[k, 1].imshow(heat[..., ::-1])
    axes[k, 0].set_title(f"{s['vehicles']} vehicles · score {s['score']} ({s['level']})"); axes[k, 1].set_title("density heatmap")
    for a in axes[k]:
        a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "results/figures/image_analysis.jpg", dpi=80); plt.show()
pd.DataFrame(rows)
""")
code("""
# Score distribution over the whole test split (sanity check of the heuristic)
scores = [an.image(cv2.imread(str(p)))[2] for p in test_imgs]
sd = pd.DataFrame(scores)
print(sd.level.value_counts().reindex(["Low", "Moderate", "Heavy", "Jam"]).fillna(0).astype(int).to_dict())
sd[["vehicles", "pcu", "occupancy", "score"]].describe().round(2)
""")

md("## 11 · Video: ByteTrack counting\nCounts each vehicle **once** (unique track IDs), counts line crossings, and tracks "
   "congestion over time. Add your own Dhaka traffic clip with *Add Input* or set `CFG['video']`. "
   "Without a video, a short **synthetic pan** over a test image is used only to check that the pipeline runs.")
code("""
def synthetic_pan(img_path, dst, seconds=8, fps=25):
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    cw, ch = int(W * 0.7), int(H * 0.7)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))
    for t in np.linspace(0, 1, seconds * fps):
        x, y = int(t * (W - cw)), int(0.5 * t * (H - ch))
        vw.write(np.ascontiguousarray(img[y:y + ch, x:x + cw]))
    vw.release()
    return str(dst)


VIDEO = CFG["video"] or next(iter(sorted(sum((glob.glob(f"/kaggle/input/**/*.{e}", recursive=True)
                                              for e in ("mp4", "avi", "mov", "mkv")), []))), "")
synthetic = not VIDEO
if synthetic:
    VIDEO = synthetic_pan(test_imgs[0], "/tmp/pan.mp4")
fps = cv2.VideoCapture(VIDEO).get(cv2.CAP_PROP_FPS) or 25
vid = an.video(VIDEO, str(OUT / "results/video_analyzed.mp4"), max_frames=int(CFG["video_seconds"] * fps))
print("source:", VIDEO, "(synthetic pan, sanity check only)" if synthetic else "")
print(json.dumps({k: v for k, v in vid.items() if k not in ("timeline", "heatmap")}, indent=1))
tl = pd.DataFrame(vid["timeline"])
fig, ax = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={"width_ratios": [2, 1]})
ax[0].plot(tl.t, tl.score, label="congestion score"); ax[0].plot(tl.t, tl.vehicles, label="vehicles in view", alpha=.6)
ax[0].set(xlabel="seconds", ylim=(0, max(100, tl.vehicles.max() + 5)), title="congestion over time"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].imshow(cv2.GaussianBlur(vid["heatmap"], (0, 0), 6), cmap="jet"); ax[1].set_title("where vehicles spent time"); ax[1].axis("off")
plt.tight_layout(); plt.savefig(OUT / "results/figures/video_timeline.png", dpi=110); plt.show()
print("annotated video ->", OUT / "results/video_analyzed.mp4")
""")

md("## 12 · Export weights: `.pt` · `.pth` · ONNX")
code("""
tag = NAME.replace("-", "_")
shutil.copy2(W / "best.pt", OUT / f"models/dhaka_{tag}.pt")                           # Ultralytics checkpoint
torch.save({"state_dict": best.model.state_dict(), "names": CLASSES, "imgsz": CFG["imgsz"]},
           OUT / f"models/dhaka_{tag}_state_dict.pth")                                # plain PyTorch weights
onnx = YOLO(str(W / "best.pt")).export(format="onnx", imgsz=CFG["imgsz"], dynamic=False, simplify=True)
shutil.move(onnx, OUT / f"models/dhaka_{tag}.onnx")
shutil.copy2(W / "best.pt", OUT / "app/dhaka_yolo11s.pt")
for p in sorted((OUT / "models").iterdir()):
    print(f"{p.name:40s} {p.stat().st_size / 1e6:6.1f} MB")
""")

md("## 13 · Gradio demo\nThe same `app/` folder can be pushed to a Hugging Face Space (`app.py` + `traffic.py` + weights).")
code("%%writefile app/app.py\n" + (ROOT / "app/app.py").read_text(encoding="utf-8").rstrip())
code("""
for f in ("app.py", "traffic.py"):
    shutil.copy2(Path("app") / f, OUT / "app" / f)
shutil.copy2("app/requirements.txt", OUT / "app/requirements.txt")
sys.modules.pop("app", None)
from app import build_demo

if CFG["demo"]:
    demo = build_demo(an)
    demo.launch(share=True, prevent_thread_lock=True)
""")

md("## 14 · Save results + zip for GitHub\nWrites the result files and packs everything the repo needs into "
   "**`results_for_github.zip`** (Output tab → download → unzip into the repo).")
code("""
import zipfile

summary = {"run": NAME, "config": CFG, "test": {k: float(v) for k, v in overall.items()},
           "per_class": per_class.reset_index().to_dict("records"),
           "data": {s: len(list((DATA / "images" / s).glob("*"))) for s in ("train", "val", "test")},
           "video": {k: v for k, v in vid.items() if k not in ("timeline", "heatmap")}, "video_synthetic": synthetic}
(OUT / "results/results.json").write_text(json.dumps(summary, indent=2, default=str))
h = pd.read_csv(RUN_DIR / "results.csv")                         # self-contained: does not need section 8's table
h.columns = [c.strip() for c in h.columns]
hc = [c for c in ("epoch", "train/box_loss", "val/box_loss", "metrics/precision(B)", "metrics/recall(B)",
                  "metrics/mAP50(B)", "metrics/mAP50-95(B)") if c in h]
hist_md = h[hc].rename(columns=lambda c: c.replace("metrics/", "").replace("(B)", "")).to_markdown(index=False, floatfmt=".4f")
with open(OUT / "results/RESULTS.md", "w") as f:
    f.write(f"## Test split ({summary['data']['test']} images)\\n\\n")
    f.write(pd.Series(overall).to_frame("value").to_markdown(floatfmt=".4f"))
    f.write("\\n\\n## Per class\\n\\n" + per_class.to_markdown(floatfmt=".3f") + "\\n")
    f.write("\\n\\n## Training history\\n\\n" + hist_md + "\\n")
shutil.copy2(RUN_DIR / "results.csv", OUT / "results/results.csv")

need = ["app/app.py", "app/traffic.py", "app/requirements.txt", "app/dhaka_yolo11s.pt", "results/RESULTS.md",
        "results/results.json", "results/results.csv"]
missing = [f for f in need if not (OUT / f).exists()]
assert not missing, f"missing: {missing} -> run sections 12-13 first"
assert any((OUT / "models").glob("*.pt")), "no model in outputs/models -> run section 12 first"

gh = Path.cwd() / "results_for_github.zip"                       # same layout as the repo
with zipfile.ZipFile(gh, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("models", "results", "app"):
        for p in sorted((OUT / sub).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
with zipfile.ZipFile(gh) as z:
    assert z.testzip() is None, "zip is corrupt"
    names = z.namelist()

for p in [p for sub in ("results", "models") for p in (OUT / sub).rglob("*") if p.is_file()] + [gh]:
    store.put(p, f"final/{p.relative_to(OUT).as_posix() if p.is_relative_to(OUT) else p.name}")
print(f"✅ {gh.name}: {len(names)} files, {gh.stat().st_size / 1e6:.1f} MB  (also in the bucket under final/)")
for folder in ("models", "results", "app"):
    print(f"   {folder + '/':9s}", ", ".join(n.split("/", 1)[1] for n in names if n.startswith(folder + "/")))
print("Unzip results_for_github.zip into the repo folder (it fills models/, results/, app/).")
print("GitHub limit is 100 MB per file: commit the .pt; keep the .onnx / .pth as a Release asset if they are larger.")
""")

def main():
    req = ROOT / "app/requirements.txt"
    C_insert = nbf.v4.new_code_cell("%%writefile app/requirements.txt\n" + req.read_text().rstrip())
    idx = next(i for i, c in enumerate(C) if c.source.startswith("%%writefile app/app.py"))
    C.insert(idx + 1, C_insert)
    nb = nbf.v4.new_notebook()
    nb.cells = C
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks/dhaka_traffic_density.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print("wrote", out, f"({len(C)} cells)")


if __name__ == "__main__":
    main()
