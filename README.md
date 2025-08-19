# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![image](https://github.com/user-attachments/assets/5988cf8a-1ac3-42a1-aa73-fd1565aceb80)



## DESIGN STEPS

### STEP 1:
Import necessary libraries.Then,the dataset is loaded.

### STEP 2:
Split the dataset into training and testing dataset.

### STEP 3:
Normalize the features,fit the model and transform the data.

### STEP 4:
Build the Neural Network Model and compile the model.Then initialize the Model, Loss Function, and Optimizer

### STEP 5:
Train the model using training data.

### STEP 6:
Plot the performance plot.

### STEP 7:
Evaluate the model with the testing data using confusion matrix, classification Report.

## PROGRAM

### Name: MOHANRAM GUNASEKAR
### Register Number: 212223240095

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # First hidden layer
        self.fc2 = nn.Linear(16, 8) # Second hidden layer
        self.fc3 = nn.Linear(8, 4) # Output layer


    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (logits)
        return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information
![image](https://github.com/user-attachments/assets/b2b12d07-9611-4bf9-b19f-471520534b59)


## OUTPUT
### Confusion Matrix
![image](https://github.com/user-attachments/assets/fb09288a-5ff7-4c97-808a-4c78fa33b949)


### Classification Report
![image](https://github.com/user-attachments/assets/14dd9610-64db-48c8-92a4-89f459fba6ce)



### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/4027b823-6ba2-463a-a8fc-464d0cda64fa)


## RESULT
Thus,the neural network classification model to predict the right group of the new customers has successfully developed.
