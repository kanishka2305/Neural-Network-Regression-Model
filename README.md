# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="773" height="527" alt="image" src="https://github.com/user-attachments/assets/51723f55-32c2-439d-b5b4-7a8298fc9a24" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Kanishka V S
### Register Number: 212222230061
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,5)
        self.fc3 = nn.Linear(5,2)
        self.fc4 = nn.Linear(2,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.fc4(x)

    return x

# Initialize the Model, Loss Function, and Optimizer

ai_thanika = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_thanika.parameters(),lr=0.001)


def train_model(ai_thanika, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_thanika(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_thanika.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information

<img width="306" height="563" alt="image" src="https://github.com/user-attachments/assets/9bbbee87-4e6c-4cdf-80f7-da4cb9834db3" />


## OUTPUT

<img width="663" height="452" alt="image" src="https://github.com/user-attachments/assets/b9a8dd3c-8771-429a-aeb8-8c5db682e043" />


### New Sample Data Prediction

<img width="733" height="117" alt="image" src="https://github.com/user-attachments/assets/c89a0218-7208-45cb-a9ec-32d2afd06850" />


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
