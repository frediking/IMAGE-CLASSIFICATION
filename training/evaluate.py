def evaluate_model(model, train_dataset, val_dataset):
    # Evaluate model performance
    train_loss, train_accuracy = model.evaluate(train_dataset)
    val_loss, val_accuracy = model.evaluate(val_dataset)
    
    print(f'\nTraining Accuracy: {train_accuracy:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    
    return train_accuracy, val_accuracy