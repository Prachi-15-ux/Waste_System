import os
import shutil

# Minimize TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

DATASET_DIR = "dataset"
EXTRACTED_DIR = os.path.join(DATASET_DIR, "dataset-resized")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "waste_model.h5")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALID_CLASSES = ['glass', 'metal', 'organic', 'paper', 'plastic']

def clean_dataset_structure(base_dir):
    """Ensures only the 5 valid classes exist in the target directory."""
    print("Cleaning dataset structure and ignoring hidden/system files...")
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist. Please place your dataset there.")
        return False
        
    # Remove unwanted files/folders at the root dataset level
    macosx_dir = os.path.join(DATASET_DIR, "__MACOSX")
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir, ignore_errors=True)
        
    # 1. Merge cardboard into paper if it exists
    cardboard_dir = os.path.join(base_dir, 'cardboard')
    paper_dir = os.path.join(base_dir, 'paper')
    if os.path.exists(cardboard_dir):
        if not os.path.exists(paper_dir):
            os.makedirs(paper_dir)
        for file in os.listdir(cardboard_dir):
            try:
                shutil.move(os.path.join(cardboard_dir, file), os.path.join(paper_dir, file))
            except Exception:
                pass
        shutil.rmtree(cardboard_dir, ignore_errors=True)
        
    # 2. Rename trash to organic if it exists
    trash_dir = os.path.join(base_dir, 'trash')
    organic_dir = os.path.join(base_dir, 'organic')
    if os.path.exists(trash_dir) and not os.path.exists(organic_dir):
        os.rename(trash_dir, organic_dir)

    # Clean up anything inside base_dir that is not in VALID_CLASSES
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if item not in VALID_CLASSES:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)

    # Ensure all required folders exist
    for target in VALID_CLASSES:
        target_dir = os.path.join(base_dir, target)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    # Print images per class
    print("\nDataset Verification:")
    total_images = 0
    for cls in VALID_CLASSES:
        cls_dir = os.path.join(base_dir, cls)
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f" - {cls}: {count} images")
        total_images += count
    print(f"Total valid images found: {total_images}\n")
    
    return True

def run_training():
    print("=== Starting Waste Classification Training Pipeline ===")
    
    success = clean_dataset_structure(EXTRACTED_DIR)
    if not success:
        return
        
    print(f"Loading dataset ONLY from: {os.path.abspath(EXTRACTED_DIR)}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        EXTRACTED_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    class_names = train_ds.class_names
    print(f"\n✅ Classes detected by TensorFlow: {class_names}")
    
    # Verify the classes are exactly what we want
    expected_classes = sorted(VALID_CLASSES)
    if sorted(class_names) != expected_classes:
        print(f"⚠️ Warning: Detected classes {class_names} do not match expected {expected_classes}")
    
    print("\nBuilding MobileNetV2 Transfer Learning Model...")
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    print("\nTraining model on REAL dataset for 3 epochs...")
    model.fit(train_ds, epochs=3)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\nSaving finalized model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Training completely finished! Model is ready for inference.")

if __name__ == "__main__":
    run_training()
