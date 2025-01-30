def calculate_accuracy(y_true, y_pred):
    """Calculate the accuracy of predictions."""
    correct_predictions = (y_true == y_pred).sum()
    accuracy = correct_predictions / len(y_true)
    return accuracy

def calculate_loss(y_true, y_pred):
    """Calculate the mean squared error loss."""
    loss = ((y_true - y_pred) ** 2).mean()
    return loss