import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from training_parts import test_images, Labels, THIS_FOLDER

# Load trained model
model_path = THIS_FOLDER + "/weights/ResNet50_BodyParts.h5"
model = load_model(model_path)

# Predict on test images
y_pred_probs = model.predict(test_images)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_images.classes

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=Labels))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Labels, yticklabels=Labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
