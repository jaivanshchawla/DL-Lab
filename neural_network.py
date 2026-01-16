import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def sigmoid(x):
    """Apply sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute derivative of sigmoid for backpropagation"""
    return x * (1 - x)

def forward_propagation(X, weights, bias):
    """Compute network output through forward propagation"""
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def train(X, y, weights, bias, epochs, learning_rate):
    """Train the network using backpropagation"""
    for epoch in range(epochs):
        # Forward pass
        output = forward_propagation(X, weights, bias)
        
        # Compute loss (MSE)
        loss = np.mean((y - output) ** 2)
        
        # Backpropagation
        error = y - output
        gradient = error * sigmoid_derivative(output)
        
        # Update weights and bias
        weights += learning_rate * np.dot(X.T, gradient)
        bias += learning_rate * np.sum(gradient, axis=0, keepdims=True)
        
        # Display progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights, bias

if __name__ == "__main__":
    # Example training data (XOR-like problem)
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Initialize weights and bias with appropriate dimensions
    # Input: 3 features, Output: 1 neuron
    weights = np.random.randn(3, 1)
    bias = np.random.randn(1, 1)
    
    print("=== Single-Layer Feedforward Neural Network ===\n")
    print("Training Data:")
    print("Inputs:\n", X)
    print("Targets:\n", y)
    print()
    
    # Show predictions before training
    print("--- Predictions Before Training ---")
    predictions_before = forward_propagation(X, weights, bias)
    print(predictions_before)
    print()
    
    # Train the network
    print("--- Training Progress ---")
    epochs = 1000
    learning_rate = 0.5
    weights, bias = train(X, y, weights, bias, epochs, learning_rate)
    print()
    
    # Show predictions after training
    print("--- Predictions After Training ---")
    predictions_after = forward_propagation(X, weights, bias)
    print(predictions_after)
    print()
    
    print("--- Final Weights and Bias ---")
    print("Weights:\n", weights)
    print("Bias:\n", bias)
