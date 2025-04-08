import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    def __init__(self, input_size, shared_hidden_size, classification_output_size, regression_output_size):
        super(MultiTaskModel, self).__init__()
        
        # Shared layers
        self.shared_layer = nn.Sequential(
            nn.Linear(input_size, shared_hidden_size),
            nn.ReLU()
        )
        
        # Task-specific layers
        self.classification_head = nn.Sequential(
            nn.Linear(shared_hidden_size, classification_output_size),
            nn.Softmax(dim=1)
        )
        
        self.regression_head = nn.Linear(shared_hidden_size, regression_output_size)

    def forward(self, x):
        # Shared representation
        shared_output = self.shared_layer(x)
        
        # Task-specific outputs
        classification_output = self.classification_head(shared_output)
        regression_output = self.regression_head(shared_output)
        
        return classification_output, regression_output

# Example usage
if __name__ == "__main__":
    # Define model
    input_size = 10
    shared_hidden_size = 32
    classification_output_size = 3  # e.g., 3 classes
    regression_output_size = 1     # e.g., single regression value

    model = MultiTaskModel(input_size, shared_hidden_size, classification_output_size, regression_output_size)
    
    # Dummy input
    x = torch.randn(5, input_size)  # Batch of 5 samples
    
    # Forward pass
    classification_output, regression_output = model(x)
    
    print("Classification Output:", classification_output)
    print("Regression Output:", regression_output)