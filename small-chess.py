import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import re

# Define the model   # a better way is to train 2 model twice in white & black perspective - one's defeat other's victory & flipping the squares
# another pass to train a another perspective where black replaced by white ?? may be  
# train a big TRAINER - to train a small network lets see 

class SquaredClampedReLU(nn.Module):  # continous & non-lineraity & differntiable -- vectorisation ? quantisation ?
    def forward(self, x):
        return torch.square(torch.clamp(x, min=0, max=1))
        
        
class LargeChessNet(nn.Module):  #  a trainer net hopefully later a small network will be made 
    def __init__(self):
        super(LargeChessNet, self).__init__()

        
        self.fc1 = nn.Linear(384, 32)  # Adjusted input size
        self.fc2 = nn.Linear(32, 1)  # Single output

        self.sq_crelu = SquaredClampedReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
     
        x = self.sq_crelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer (no activation)

        return x
        
        
# FEN to tensor conversion (one-hot-like encoding)  # need to add castling & enpassante - 4+16
def fen_to_tensor(fen):
    pieces = {'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6,
              'P': 7, 'R': 8, 'N': 9, 'B': 10, 'Q': 11, 'K': 12}
    board = np.zeros(64, dtype=np.int8)  # 64 digit array 
    parts = fen.split()
    rows = parts[0].split('/')
    multiplier = 1 
    enemy = -1  # training a big network no needed 
    friend = 1 
    stm = 0
    if (parts[1] == "black"):  # only changing the perspective 
        stm = 1  
        multiplier = -1 
    #    print("flipping")      # will flip the board 
    square_index = 0
  #  print(rows)                # prints the FEN 
    for row in rows:
      #  print(row)                  # prints each part  
        for piece in row:
            if piece.isdigit():            # if number return empty 0s 
                square_index += int(piece)
            else:
                board[square_index] = pieces.get(piece, 0)
                square_index += 1
    
    one_hot = np.zeros((6, 64),dtype=np.float32)  # simple network 
    
    for i in range(64):
                                                      
      if board[i]!=0:                   
         # storing the data   # flipping the boxes if black 
        if(board[i]<7):
            one_hot[board[i]-1][(i^(56*stm))]=friend*multiplier   # friendly ==1  # piece*row*channel  
        if (board[i]>6):
            one_hot[board[i]-7][i^(56*stm)]= enemy*multiplier # enemy == -1 although not needed ig i will remove 
              
    return torch.tensor(one_hot.flatten(), dtype=torch.float32) # .flatten() no since convulated 

# Custom Dataset
class ChessDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    fen, score = line.strip().split(';')  # Split by semicolon
                    self.data.append((fen, float(score))) #store as tuple
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, score = self.data[idx]
        fen_tensor = fen_to_tensor(fen)
        
        sc_bl = fen.split()
        if (sc_bl[1]=="black"):
            score = 1- score 
         #   print("score reversed")
            
        return fen_tensor, torch.tensor([score], dtype=torch.float32)

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        for fen_tensor, score_tensor in train_loader:
            fen_tensor = fen_tensor.to(device)
            score_tensor = score_tensor.to(device)

            optimizer.zero_grad()
            outputs = model(fen_tensor)
            loss = criterion(outputs, score_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * fen_tensor.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()   # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation during validation
            for fen_tensor, score_tensor in val_loader:
                fen_tensor = fen_tensor.to(device)
                score_tensor = score_tensor.to(device)
                outputs = model(fen_tensor)
                loss = criterion(outputs, score_tensor)
                val_loss += loss.item() * fen_tensor.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for fen_tensor, score_tensor in test_loader:
            fen_tensor = fen_tensor.to(device)
            score_tensor = score_tensor.to(device)
            outputs = model(fen_tensor)
            loss = criterion(outputs, score_tensor)
            test_loss += loss.item() * fen_tensor.size(0)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')



#with open("C:/Users/vaibhav/Desktop/VIJAY_SLOWBOT/ext_data/quiet-labeled.epd", "r") as file:


full_dataset = ChessDataset("C:/Users/vaibhav/Desktop/VIJAY_SLOWBOT/ext_data/quiet-labeled.epd")

train_size = int(0.8 * len(full_dataset))  # 80% for training
val_size = len(full_dataset) - train_size      # 20% for validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4096) # No need to shuffle validation set

test_dataset = ChessDataset("C:/Users/vaibhav/Desktop/VIJAY_SLOWBOT/ext_data/quiet-labeled2.epd")
test_loader = DataLoader(test_dataset, batch_size=4096)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LargeChessNet().to(device)


criterion = nn.L1Loss() #nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 5

train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device)

test_model(model,test_loader,nn.MSELoss(),device)

torch.save(model.state_dict(), "vijay1_small_model.pth")
