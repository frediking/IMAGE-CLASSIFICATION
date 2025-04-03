
def train_model(model, train_dataset, val_dataset, epochs=20):
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset
    )
    return history

