import os, boto3
from huggingface_hub import hf_hub_download

BUCKET = "worldguessr-ml-data"
REPO = "osv5m/osv5m"
NUM_TRAIN_SHARDS = 10

shards = [f"images/test/{i:02d}.zip" for i in range(5)]
shards += [f"images/train/{i:02d}.zip" for i in range(NUM_TRAIN_SHARDS)]

s3 = boto3.client("s3")
for f in shards:
    key = f"raw/{f}"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        print("already in S3, skipping:", key, flush=True)
        continue
    except s3.exceptions.ClientError:
        pass
    path = hf_hub_download(REPO, filename=f, repo_type="dataset", local_dir="data/raw")
    print("uploading", key, flush=True)
    s3.upload_file(path, BUCKET, key)
    os.remove(path)
    print("done", key, flush=True)
print("ALL DONE", flush=True)
