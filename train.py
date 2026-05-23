import os
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PATCHES_PATH = os.path.join(_BASE_DIR, "patches")
MODEL_SAVE   = os.path.join(_BASE_DIR, "unet_model.keras")
PATCH_SIZE   = 256
NUM_CLASSES  = 6
BATCH_SIZE   = 8
EPOCHS       = 30

print("GPUs available:", tf.config.list_physical_devices('GPU'))

# ── Load file paths ─────────────────────────────────────────────
img_paths  = sorted(Path(PATCHES_PATH, "images").glob("*.png"))
mask_paths = sorted(Path(PATCHES_PATH, "masks").glob("*.npy"))
print(f"Total patches: {len(img_paths)}")

# ── Train / Validation split ────────────────────────────────────
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    img_paths, mask_paths, test_size=0.2, random_state=42
)
print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

# ── Class weights (to handle imbalance) ────────────────────────
# Land=53% is dominant, so we up-weight rare classes
CLASS_WEIGHTS = np.array([2.0, 0.5, 2.5, 2.5, 2.0, 0.1], dtype=np.float32)

# ── Data pipeline ───────────────────────────────────────────────
def load_pair(img_path, mask_path):
    img  = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.load(mask_path).astype(np.int32)
    return img, mask

def augment(img, mask):
    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask[..., tf.newaxis])[..., 0]
    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask[..., tf.newaxis])[..., 0]
    # Random brightness
    img = tf.image.random_brightness(img, 0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, mask

def make_dataset(img_paths, mask_paths, augment_data=False, batch_size=BATCH_SIZE):
    def generator():
        for ip, mp in zip(img_paths, mask_paths):
            yield load_pair(str(ip), str(mp))

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE),    dtype=tf.int32),
        )
    )
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(200).batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_imgs, train_masks, augment_data=True)
val_ds   = make_dataset(val_imgs,   val_masks,   augment_data=False)

# ── U-Net architecture ──────────────────────────────────────────
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    return x

def encoder_block(x, filters):
    skip = conv_block(x, filters)
    pool = layers.MaxPooling2D(2)(skip)
    return skip, pool

def decoder_block(x, skip, filters):
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(256, 256, 3), num_classes=6):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1,     64)
    s3, p3 = encoder_block(p2,     128)
    s4, p4 = encoder_block(p3,     256)

    # Bottleneck
    b = conv_block(p4, 512)

    # Decoder
    d1 = decoder_block(b,  s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(d4)
    return Model(inputs, outputs, name="U-Net")

model = build_unet()
model.summary()

# ── Loss with class weights ─────────────────────────────────────
def weighted_sparse_cce(y_true, y_pred):
    weights = tf.constant(CLASS_WEIGHTS)
    loss    = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    w_map   = tf.gather(weights, tf.cast(y_true, tf.int32))
    return tf.reduce_mean(loss * w_map)

# ── Compile ─────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=weighted_sparse_cce,
    metrics=["accuracy"]
)

# ── Callbacks ───────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE, save_best_only=True,
        monitor="val_loss", mode="min", verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=4, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True, verbose=1
    ),
]

# ── Train ───────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Plot training curves ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train loss")
axes[0].plot(history.history["val_loss"], label="Val loss")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["accuracy"],     label="Train acc")
axes[1].plot(history.history["val_accuracy"], label="Val acc")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(_BASE_DIR, "training_curves.png"))
plt.show()
print("\n✅ Training complete! Model saved.")