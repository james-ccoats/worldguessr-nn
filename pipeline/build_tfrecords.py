"""Convert OSV-5M zip shards in S3 into 224x224 TFRecords.
Each record stores: JPEG bytes, country code (as text), image id.
Country filtering happens at training time, not here."""
import io, os, json, zipfile #reads shards without extracting them to disk
from collections import Counter #tally images per country
from multiprocessing import Pool #runs multiple shards at once
import boto3, pandas as pd #talk to s3, read csvs
from PIL import Image #decodes and resizes images

BUCKET = "worldguessr-ml-data"
SIZE = 224 #native output resolution of CLIP
OUT_PREFIX = f"processed/tfrecords-{SIZE}" 
LOCAL_OUT = f"data/processed/tfrecords-{SIZE}"
WORK = "data/work" #scratch folder for processing
SHARDS = [("test", i) for i in range(5)] + [("train", i) for i in range(10)]

# Loaded once in the parent; forked workers share it
META = {}
for split in ("train", "test"):
    df = pd.read_csv(f"data/raw/{split}.csv", usecols=["id", "country"], dtype={"id": str})
    META[split] = dict(zip(df.id, df.country))
    print(f"loaded {split} metadata: {len(df)} rows", flush=True)

def process_shard(job):
    import tensorflow as tf  # imported inside the worker
    split, idx = job
    s3 = boto3.client("s3")
    name = f"shard-{idx:02d}"
    rec_key = f"{OUT_PREFIX}/{split}/{name}.tfrecord"
    cnt_key = f"{OUT_PREFIX}/{split}/{name}.counts.json"

    try:  # already done on a previous run
        s3.head_object(Bucket=BUCKET, Key=rec_key)
        body = s3.get_object(Bucket=BUCKET, Key=cnt_key)["Body"].read()
        print(f"skip {split}/{name} (already in S3)", flush=True)
        return split, json.loads(body)
    except s3.exceptions.ClientError:
        pass

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(f"{LOCAL_OUT}/{split}", exist_ok=True)
    zip_path = f"{WORK}/{split}-{idx:02d}.zip"
    rec_path = f"{LOCAL_OUT}/{split}/{name}.tfrecord"
    s3.download_file(BUCKET, f"raw/images/{split}/{idx:02d}.zip", zip_path)

    def b(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    counts, bad, nolabel = Counter(), 0, 0
    with zipfile.ZipFile(zip_path) as z, tf.io.TFRecordWriter(rec_path) as w:
        for member in z.namelist():
            if not member.lower().endswith(".jpg"):
                continue
            img_id = os.path.basename(member)[:-4]
            country = META[split].get(img_id)
            if country is None:
                nolabel += 1
                continue
            try:
                # Squash to a square (keeps the whole scene, slight aspect distortion)
                img = Image.open(io.BytesIO(z.read(member))).convert("RGB")
                img = img.resize((SIZE, SIZE), Image.BICUBIC)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
            except Exception:
                bad += 1
                continue
            ex = tf.train.Example(features=tf.train.Features(feature={
                "image": b(buf.getvalue()),
                "country": b(country.encode()),
                "id": b(img_id.encode()),
            }))
            w.write(ex.SerializeToString())
            counts[country] += 1

    s3.upload_file(rec_path, BUCKET, rec_key)
    s3.put_object(Bucket=BUCKET, Key=cnt_key, Body=json.dumps(counts).encode())
    os.remove(zip_path)
    print(f"done {split}/{name}: {sum(counts.values())} images, "
          f"{bad} unreadable, {nolabel} without label", flush=True)
    return split, dict(counts)

if __name__ == "__main__":
    totals = {"train": Counter(), "test": Counter()}
    with Pool(4) as pool:
        for split, counts in pool.imap_unordered(process_shard, SHARDS):
            totals[split].update(counts)

    os.makedirs("data/processed", exist_ok=True)
    for split, c in totals.items():
        with open(f"data/processed/counts_{split}.json", "w") as f:
            json.dump(dict(c.most_common()), f, indent=1)
        print(f"{split}: {sum(c.values())} images, {len(c)} countries", flush=True)

    tr = totals["train"]
    for n in [50, 100, 200, 500]:
        kept = [k for k, v in tr.items() if v >= n]
        test_kept = sum(totals["test"][k] for k in kept)
        print(f"threshold {n}: {len(kept)} countries, "
              f"{test_kept}/{sum(totals['test'].values())} test images kept", flush=True)
    print("ALL DONE", flush=True)
