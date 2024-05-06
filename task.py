import csv
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        data = [row for row in reader]
    return np.array(data, dtype=int)

def plot_digits(data, num_digits=10):
    _, axes = plt.subplots(1, num_digits, figsize=(10, 2))
    for ax, row in zip(axes, data[:num_digits]):
        img = row[1:].reshape(28, 28)  # The first element is the label, the rest are pixels
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {row[0]}')
        ax.axis('off')
    plt.show()

def train_naive_bayes(train_data):
    labels = train_data[:, 0]
    features = train_data[:, 1:]
    num_classes = len(np.unique(labels))
    num_features = features.shape[1]

    # Initialize probability dictionary
    probabilities = {
        'class_prob': np.zeros(num_classes),
        'pixel_prob': np.zeros((num_classes, num_features, 256))  # 256 for pixel values
    }

    # Calculate the probability for each class
    for c in range(num_classes):
        class_data = features[labels == c]
        probabilities['class_prob'][c] = len(class_data) / len(features)

        # Calculate the probability for each pixel value with Laplace smoothing
        for i in range(num_features):
            pixel_values = class_data[:, i]
            pixel_counts = np.zeros(256)
            for value in pixel_values:
                pixel_counts[value] += 1
            probabilities['pixel_prob'][c, i, :] = (pixel_counts + 1) / (len(class_data) + 256)

    return probabilities

def predict(test_data, probabilities):
    num_classes = len(probabilities['class_prob'])
    num_features = test_data.shape[1]
    predictions = []
    
    for instance in test_data:
        class_scores = np.log(probabilities['class_prob'])  # start with the logarithm of class probabilities
        for c in range(num_classes):
            pixel_log_probs = np.log(probabilities['pixel_prob'][c, range(num_features), instance])
            class_scores[c] += np.sum(pixel_log_probs)
        
        predicted_class = np.argmax(class_scores)
        predictions.append(predicted_class)
    return predictions

def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)

train_data = load_data('train.csv')
# plot_digits(train_data)
test_data = load_data('test.csv')
# plot_digits(test_data)

probabilities = train_naive_bayes(train_data)

test_features = test_data[:, 1:]
test_labels = test_data[:, 0]

predictions = predict(test_features, probabilities)

accuracy = calculate_accuracy(predictions, test_labels)

print(f"Model accuracy: {accuracy * 100:.2f}%")
