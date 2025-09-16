# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="631" height="968" alt="image" src="https://github.com/user-attachments/assets/c6e04623-886f-4b95-af9a-14cd7fadc006" />


## DESIGN STEPS

### STEP 1
Load and preprocess the dataset (handle missing values, encode categorical features, scale numeric data).

### STEP 2
Split the dataset into training and testing sets, convert to tensors, and create DataLoader objects.

### STEP 3
Build the neural network model, train it with CrossEntropyLoss and Adam optimizer, then evaluate with confusion matrix and classification report.

## PROGRAM
### Name: Mohanram Gunasekar
### Register Number: 212223240095

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        # Simple 3-layer fully connected network
        self.fc1 = nn.Linear(input_size, 64)   # Input -> 64
        self.fc2 = nn.Linear(64, 32)           # 64 -> 32
        self.fc3 = nn.Linear(32, 4)            # 32 -> 4 classes (A,B,C,D)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        # raw logits; CrossEntropyLoss applies Softmax
        return x
```
```python
input_size = X_train.shape[1]
model = PeopleClassifier(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader, criterion, optimizer, epochs=50)
```

## Dataset Information

<img width="912" height="725" alt="image" src="https://github.com/user-attachments/assets/edbbcc9d-3fc5-41d4-8c5b-0b5d6ccdda93" />


## OUTPUT

### Confusion Matrix



<img width="278" height="184" alt="image" src="https://github.com/user-attachments/assets/de2d83ba-089b-49d5-a733-39b068266da8" />






<img width="714" height="573" alt="image" src="https://github.com/user-attachments/assets/48a2ce6a-7160-4924-a1f3-872de39083ae" />




### Classification Report



<img width="613" height="270" alt="image" src="https://github.com/user-attachments/assets/fba6c413-0b88-4c07-bc23-df184533bf4d" />




### New Sample Data Prediction



<img width="369" height="84" alt="image" src="https://github.com/user-attachments/assets/ff9bdd49-5292-4b12-97a3-0d19e88032e2" />




## RESULT
The neural network model was successfully built and trained to handle classification tasks.
