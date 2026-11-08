"""Multinomial logistic regression (single Dense + softmax) on 64x64 raw pixels."""
import json, os, glob
import boto3, tensorflow as tf

BUCKET = "worldguessr-ml-data"
MIN_COUNT = 100       # keep countries with >= this many training images
IMG = 64              # downsample 224 -> 64
BATCH = 512
EPOCHS = 15
LR = 1e-3
EVAL_TEST = True      # set False while tuning; True for final numbers

DATA = "data/processed"
REC = f"{DATA}/tfrecords-224"
RUN = f"data/runs/logreg_{IMG}px_min{MIN_COUNT}"
os.makedirs(RUN, exist_ok=True)

#Label mapping, shared with the LoRA model later
counts_train = json.load(open(f"{DATA}/counts_train.json"))
counts_test = json.load(open(f"{DATA}/counts_test.json"))
countries = sorted(c for c, n in counts_train.items() if n >= MIN_COUNT)
c2i = {c: i for i, c in enumerate(countries)}
map_path = f"{DATA}/country_to_idx_min{MIN_COUNT}.json"
json.dump(c2i, open(map_path, "w"), indent=1)
boto3.client("s3").upload_file(map_path, BUCKET, f"processed/country_to_idx_min{MIN_COUNT}.json")
NUM_CLASSES = len(countries)
print(f"{NUM_CLASSES} classes", flush=True)

table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(countries), tf.range(NUM_CLASSES, dtype=tf.int64)),
    default_value=-1)

FEATURES = {"image": tf.io.FixedLenFeature([], tf.string),
            "country": tf.io.FixedLenFeature([], tf.string)}

def parse(record):
    f = tf.io.parse_single_example(record, FEATURES)
    label = table.lookup(f["country"])
    img = tf.io.decode_jpeg(f["image"], channels=3)
    img = tf.image.resize(img, (IMG, IMG), antialias=True)
    return tf.cast(tf.round(img), tf.uint8), label   # uint8 keeps the cache small

def make_ds(files, cache_name=None, shuffle=False):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, y: y >= 0)                # drop countries below cutoff
    if cache_name:
        ds = ds.cache(f"{RUN}/cache_{cache_name}")
    if shuffle:
        ds = ds.shuffle(50_000)
    ds = ds.batch(BATCH).map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.prefetch(tf.data.AUTOTUNE)

train_files = sorted(glob.glob(f"{REC}/train/shard-0[0-8].tfrecord"))
val_files = [f"{REC}/train/shard-09.tfrecord"]
test_files = sorted(glob.glob(f"{REC}/test/*.tfrecord"))

train_ds = make_ds(train_files, "train", shuffle=True)
val_ds = make_ds(val_files, "val")

# Model: normalize pixels, flatten, one Dense softmax layer
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(make_ds(train_files[:1]).take(40).map(lambda x, y: x))  # ~20k images

model = tf.keras.Sequential([
    tf.keras.Input((IMG, IMG, 3)),
    norm,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
)
model.summary()

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(f"{RUN}/history.csv"),
    ],
)
model.save(f"{RUN}/model.keras")

# Results
majority = max(countries, key=lambda c: counts_train[c])
test_kept = sum(counts_test.get(c, 0) for c in countries)
results = {"majority_class": majority,
           "majority_baseline_test_acc": counts_test.get(majority, 0) / test_kept}
if EVAL_TEST:
    loss, acc, top5 = model.evaluate(make_ds(test_files))
    results.update(test_loss=loss, test_acc=acc, test_top5=top5)
print(json.dumps(results, indent=1), flush=True)
json.dump(results, open(f"{RUN}/results.json", "w"), indent=1)

s3 = boto3.client("s3")
for name in ["model.keras", "history.csv", "results.json"]:
    s3.upload_file(f"{RUN}/{name}", BUCKET, f"runs/{os.path.basename(RUN)}/{name}")
print("ALL DONE", flush=True)
