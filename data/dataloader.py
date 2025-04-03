
# Preprocessing function to resize and normalize images
def load_and_preprocess_data(dataset_name='cifar10', img_size=(224, 224), batch_size=32):

    # Load dataset
    dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    train_ds, test_ds = dataset['train'], dataset['test']

    # Apply preprocessing
    train_ds = train_ds.map(preprocess).shuffle(1000).batch(batch_size)
    test_ds = test_ds.map(preprocess).batch(batch_size)

    # Enable prefetching
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, info