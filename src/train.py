import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

def train_model():
    # 1. CONFIGURATION
    data_dir = 'data'

    # Data preprocessing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Checking dataset...")

    # 2. DATA LOADING
    try:
        # Load data from folders
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        
        # Create dataloaders
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=0)
                      for x in ['train', 'val']}
    
        class_names = image_datasets['train'].classes
        print(f"Success! Classes found: {class_names}")
        print(f"Training images: {len(image_datasets['train'])}")
    
    except FileNotFoundError:
        print(f"Error: Could not find '{data_dir}' directory.")
        return
    except Exception as e:
        print(f"Unexpected error: {e} ")
        return

    # 3. MODEL PREPARATION
    print("Downloading Model (ResNet18)... ")
    model = models.resnet18(pretrained=True)

    # Change the final layer for 2 classes (Defect vs Normal)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. TRAINING LOOP
    print("Starting training... (This may take a while)")
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
    # 5. SAVE MODEL
    torch.save(model.state_dict(), 'defect_model.pth')
    print("\nTraining complete. Model saved as 'defect_model.pth'")

if __name__ == '__main__':
    train_model()

