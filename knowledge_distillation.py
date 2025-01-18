import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define SquaredClampedReLU (same as before)
class SquaredClampedReLU(nn.Module):
    def forward(self, x):
        return torch.square(torch.clamp(x, min=0, max=1))


# will use stockfish as convulated layers only have marginal improvements 
# Teacher Model (Vijay.pth - 3 Conv + 2 FC)
# class TeacherNet(nn.Module):
    # def __init__(self):
        # super(TeacherNet, self).__init__()
        # self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.fc1 = nn.Linear(128 * 8 * 8, 512)
        # self.fc2 = nn.Linear(512, 1)
        # self.sq_crelu = SquaredClampedReLU()
        # self.dropout = nn.Dropout(0.3)

    # def forward(self, x):
        # x = self.sq_crelu(self.bn1(self.conv1(x)))
        # x = self.dropout(x)
        # x = self.sq_crelu(self.bn2(self.conv2(x)))
        # x = self.dropout(x)
        # x = self.sq_crelu(self.bn3(self.conv3(x)))
        # x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        # x = self.sq_crelu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        # return x

# Student Model (3 Layers - 384, 32, 1)
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(384, 32)  # Adjusted input size
        self.fc2 = nn.Linear(32, 1)  # Single output

        self.sq_crelu = SquaredClampedReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
     
        x = self.sq_crelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer (no activation)

        return x


def distill_loss(student_logits, teacher_logits, temperature):
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(student_probs, teacher_probs) * (temperature**2)


def train_distill(teacher, student, train_loader, optimizer, criterion, temperature, alpha, num_epochs, device):
    teacher.eval() # teacher in eval mode
    student.to(device)
    for epoch in range(num_epochs):
        student.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            student_loss = criterion(student_logits, labels)
            distillation_loss = distill_loss(student_logits, teacher_logits, temperature)

            loss = alpha * distillation_loss + (1 - alpha) * student_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = TeacherNet().to(device)
teacher_model.load_state_dict(torch.load("vijay.pth"))  # Load teacher's weights
student_model = StudentNet().to(device)

# Example data loaders (replace with your actual data loaders)
# train_loader = ...

criterion = nn.MSELoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

temperature = 5.0
alpha = 0.5
num_epochs = 10

train_distill(teacher_model, student_model, train_loader, optimizer, criterion, temperature, alpha, num_epochs, device)
torch.save(student_model.state_dict(), "distilled_student.pth")