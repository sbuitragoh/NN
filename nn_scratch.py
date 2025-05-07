import numpy as np
from typing import Tuple, List

rng = np.random.default_rng(seed = 42)
CLASS_MAPPING = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

def transform_iris_data() -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    
    with open('iris_data.txt', 'r') as f:
        data : List[str] = [line.strip().split(',') for line in f.readlines()]

    for row in data[:-1]:
        features : List[float] = list(map(float, row[:-1]))
        label : int = CLASS_MAPPING[row[-1]]
        X.append(features)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1) 

    return X, y

def split_data(X : np.ndarray, y : np.ndarray, train_size : float = 0.6, val_size : float = 0.2)-> \
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle the data
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    # Split the data
    train_end : int = int(train_size * len(indices))
    val_end : int = train_end + int(val_size * len(indices))

    train_indices, val_indices, test_indices = np.split(indices, [train_end, val_end])

    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

def sigmoid(x : np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x : np.ndarray) -> np.ndarray:
    return x * (1 - x)

class NeuralNetwork:
    '''type definitions'''
    weights_input_hidden1: np.ndarray
    weights_hidden1_hidden2: np.ndarray
    weights_hidden2_output: np.ndarray
    bias_hidden1: np.ndarray
    bias_hidden2: np.ndarray
    bias_output: np.ndarray
    hidden_layer1_output: np.ndarray
    hidden_layer2_output: np.ndarray
    output: np.ndarray

    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int):
        def init_const(from_: int, to_: int) -> Tuple[np.ndarray, np.ndarray]:
            return rng.random((from_, to_)), rng.random(to_)
        
        self.weights_input_hidden1, self.bias_hidden1 = init_const(input_size, hidden_size1)
        self.weights_hidden1_hidden2, self.bias_hidden2 = init_const(hidden_size1, hidden_size2)
        self.weights_hidden2_output, self.bias_output = init_const(hidden_size2, output_size)

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.hidden_layer1_output = sigmoid(np.dot(X, self.weights_input_hidden1) + self.bias_hidden1)
        self.hidden_layer2_output = sigmoid(np.dot(self.hidden_layer1_output, self.weights_hidden1_hidden2) + self.bias_hidden2)
        self.output = sigmoid(np.dot(self.hidden_layer2_output, self.weights_hidden2_output) + self.bias_output)
        return self.output

    def backpropagation(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden2_error = np.dot(output_delta, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * sigmoid_derivative(self.hidden_layer2_output)

        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * sigmoid_derivative(self.hidden_layer1_output)

        self.weights_hidden2_output += np.dot(self.hidden_layer2_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate

        self.weights_hidden1_hidden2 += np.dot(self.hidden_layer1_output.T, hidden2_delta) * learning_rate
        self.bias_hidden2 += np.sum(hidden2_delta, axis=0) * learning_rate

        self.weights_input_hidden1 += np.dot(X.T, hidden1_delta) * learning_rate
        self.bias_hidden1 += np.sum(hidden1_delta, axis=0) * learning_rate

    def train(self, X: np.ndarray,
               y: np.ndarray, 
               X_val: np.ndarray, 
               y_val: np.ndarray, 
               epochs: int, 
               learning_rate: float) -> None:
        
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 10 == 0:
                loss = np.mean((y - self.output) ** 2)
                accuracy = self.forward(X_val, y_val)
                print(f"Epoch {epoch}, Loss: {loss}, Validation Accuracy: {accuracy * 100:.2f}%")

    def forward(self, X_val: np.ndarray, 
                y_val: np.ndarray, 
                mode: str = 'val') -> float | int:
        
        predictions = self.feedforward(X_val)
        predictions = np.clip(predictions, 0, 1)
        predicted_classes = np.argmax(predictions, axis=1)
        if mode != 'val':
            return predicted_classes
        true_classes = y_val.flatten()
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

def main():
    X, y = transform_iris_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    nn = NeuralNetwork(input_size=4, hidden_size1=8, hidden_size2=8, output_size=3)

    X_train = X_train / np.max(X_train, axis=0)
    X_val = X_val / np.max(X_val, axis=0)
    X_test = X_test / np.max(X_test, axis=0)

    y_train_one_hot = np.eye(y_train.max() + 1)[y_train.flatten()]

    nn.train(X_train, y_train_one_hot, X_val, y_val, epochs=500, learning_rate=1e-1)

    accuracy = nn.forward(X_test, y_test)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

    while True:
        user_input = input("Enter flower features (sepal length, sepal width, petal length, petal width) or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        try:
            features = np.array(list(map(float, user_input.split(',')))).reshape(1, -1)
            features = features / np.max(features, axis=1)
            pred = nn.forward(features, None, mode = 'test')[0]
            print(f"Predicted class name: {list(CLASS_MAPPING.keys())[pred]}")
        except Exception as e:
            print(f"Error: {e}. Please enter valid numeric values.")

if __name__ == "__main__":
    main()