

"""
Video explanation: https://www.youtube.com/watch?v=VJW9wU-1n18
Starting code came from: https://github.com/patrickloeber/pytorchTutorial/blob/master/13_feedforward.py
"""
import sys 
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Where to output the logs, so they can also be re-used when started up again
writer = SummaryWriter('runs/mnist')


# Device configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# New way of loading: https://nextjournal.com/gkoehler/pytorch-mnist
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')

# Grid to display examples in
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images_batch64', img_grid)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


# Creates an tracks a visual graph by passing 1 batch of data so it can track each operation
# Now under 'graphs' you can see each layer in the model and you can easily inspect them
writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))

# # Makes sure all the outputs are being flushed
# writer.close()
# sys.exit()

# Metrics to keep track of for the tensorboard
running_loss = 0.0
running_correct = 0

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        running_loss += loss.item()
        
        # Predicted, max the probs
        _, predicted = torch.max(outputs.data, 1)
        
        # How many are correct
        running_correct += (predicted == labels).sum().item()

        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            # Because for each 100 steps summed up
            writer.add_scalar('training loss', running_loss/100, epoch *n_total_steps + i)
            writer.add_scalar('training accuracy', running_correct/100, epoch *n_total_steps + i)
            
            # Set to 0 again for next 100
            running_loss = 0.0
            running_correct = 0


class_labels = []
class_preds = []

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# Generate precision recall curves for each of the 10 labels
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # Need sofmax because not linear output layer or softmax
        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

        # Used to calculate p and r for each class
        class_preds.append(class_probs_batch)
        class_labels.append(predicted)
        

    # 10000, 10, and 10000, 1
    # stack concatenates tensors along a new dimension
    # cat concatenates tensors in the given dimension
    # For each of the 10 classes, we stack it 
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    
    # 10 digits to loop for to get the precision and recall for this class
    classes = range(10)
    
    # Writing to tensorboard for the Precision Recall curve
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        
        # Name first and then the target and then the pred value
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()