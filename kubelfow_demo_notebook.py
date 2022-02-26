import torch
import os
import dill
import torch
import mlflow
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from mlflow import log_metric, log_param, log_artifact

print('hi')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = pd.read_csv('StudentsPerformance.csv')
feature_X = data.iloc[:, :5]
target = data['math score']


ohe = OneHotEncoder()
ohe.fit(feature_X)
feature_encoded = ohe.transform(feature_X)
feature_encoded = pd.DataFrame(feature_encoded.toarray())


# https://towardsdatascience.com/pytorch-tabular-regression-428e9c9ac93
from sklearn.model_selection import train_test_split
target = target/target.max()

# training test split
X_train, X_test, y_train, y_test = train_test_split(feature_encoded, target, test_size=0.10, random_state=1)

# Separate 10% of training data as validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)


from torch.utils.data import Dataset, DataLoader

class InputData(Dataset):
    
    def __init__(self, features, target):
        self.features = features
        self.target   = target
        
    def __getitem__(self, index):
        return self.features[index], self.target[index]
        
    def __len__ (self):
        return len(self.features)

train_dataset = InputData(torch.from_numpy(X_train.values).float(), torch.from_numpy(y_train.values).float())
val_dataset   = InputData(torch.from_numpy(X_val.values).float(),   torch.from_numpy(y_val.values).float())
test_dataset  = InputData(torch.from_numpy(X_test.values).float(),  torch.from_numpy(y_test.values).float())


epochs        = 150
batch_size    = 32
learning_rate = 0.001
feature_size  = len(X_train.columns)


train_loader = DataLoader(dataset  = train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(dataset  = val_dataset,   batch_size=1)
test_loader  = DataLoader(dataset  = test_dataset,  batch_size=1)


class Regression(nn.Module):
    def __init__(self, num_features):
        super(Regression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 10)
        self.out = nn.Linear(10, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.out(x)
        return (x)
    



import torch.optim as optim

model = Regression(feature_size)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


loss_stats = {
    'train': [],
    "val": []}

epochs = 200


from tqdm.notebook import tqdm


print("Training started !!!")
for epoch in range(1, epochs+1):
  
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
      
        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
      
        train_loss.backward()
        optimizer.step()
        
    # validation    
    with torch.no_grad():
        
         model.eval()
         for X_val_batch, y_val_batch in val_loader:
             X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
             y_val_pred = model(X_val_batch)
                        
             val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

    valIdation_loss = float(val_loss.detach().numpy())
    training_loss   = float(train_loss.detach().numpy())
    print(valIdation_loss, training_loss)

    log_metric(key="val_loss", value=valIdation_loss, step=epoch) 
    log_metric(key="train_loss", value=training_loss, step=epoch) 

X, y = next(iter(test_loader))
print(f"Model input: {X.size()}")
torch_out = model(X.to("cpu"))
print(f"Model output: {torch_out.detach().cpu().size()}")




print("Training finished !!!")

# Performance on test set

print("Model Evaluation Starts here !!!")

y_pred_list = []
with torch.no_grad():
     model.eval()
     for X_batch, _ in test_loader:
         X_batch = X_batch.to(device)
         y_test_pred = model(X_batch)
         y_pred_list.append(y_test_pred.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# Calculate MAE between predicted and target values in test set

from sklearn.metrics import mean_absolute_error

mae_DNN = mean_absolute_error(y_pred_list, y_test)*100
print(mae_DNN)
