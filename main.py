import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.backends.cudnn as cudnn

# Enable optimizations
cudnn.benchmark = True
torch.backends.cudnn.enabled = True



class FER2013Dataset(Dataset):
    """Dataset class for FER2013 emotion recognition dataset"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Map emotion categories to labels
        self.emotion_map = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5,
            'surprise': 6
        }
        
        # Load images and labels
        for emotion in self.emotion_map.keys():
            emotion_dir = os.path.join(data_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_file in os.listdir(emotion_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(emotion_dir, img_file)
                        self.images.append(img_path)
                        self.labels.append(self.emotion_map[emotion])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image with error handling
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image fails to load, create a black image
            print(f"Warning: Failed to load {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class EmotionRecognizer(nn.Module):
    """MobileNetV2-based emotion recognition model"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super(EmotionRecognizer, self).__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Replace the classifier
        # MobileNetV2's classifier has 1280 input features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', 
                learning_rate=0.001, save_path='best_model.pth', use_amp=True):
    """Train the emotion recognition model with mixed precision training"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == 'cuda') else None
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.6f}\n')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f'✓ Saved best model with validation accuracy: {val_acc:.2f}%\n')
    
    return train_losses, val_accuracies


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model on test set"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Testing')
        for images, labels in test_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f'\nTest Accuracy: {accuracy * 100:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=emotion_names))
    
    print('\nConfusion Matrix:')
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    return accuracy


def main():
    # Configuration
    DATA_DIR = 'FER2013'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    BATCH_SIZE = 64  # Increased batch size for better GPU utilization
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    # Optimize num_workers based on CPU cores
    NUM_WORKERS = min(8, os.cpu_count() or 1)  # Use up to 8 workers or available CPU cores
    IMG_SIZE = 224
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'  # Use mixed precision on GPU
    
    print(f'Using device: {device}')
    print(f'Mixed precision training: {use_amp}')
    print(f'Data loader workers: {NUM_WORKERS}')
    print(f'Batch size: {BATCH_SIZE}\n')
    
    # Data transformations - optimized for speed
    # Training: augmentation + normalization
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test: only normalization
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print('Loading datasets...')
    # Create full training dataset to get the size for splitting
    full_train_dataset = FER2013Dataset(TRAIN_DIR, transform=train_transform)
    
    # Split train into train and validation (80-20 split) using indices
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Generate random indices for splitting
    indices = np.arange(dataset_size)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets with appropriate transforms
    train_dataset_full = FER2013Dataset(TRAIN_DIR, transform=train_transform)
    val_dataset_full = FER2013Dataset(TRAIN_DIR, transform=val_test_transform)
    
    # Create subsets using indices
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    test_dataset = FER2013Dataset(TEST_DIR, transform=val_test_transform)
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}\n')
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None
    )
    
    # Create model
    print('Creating MobileNetV2 model...')
    model = EmotionRecognizer(num_classes=7, pretrained=True)
    model = model.to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    if hasattr(torch, 'compile') and device.type == 'cuda':
        print('Compiling model for faster training...')
        model = torch.compile(model, mode='reduce-overhead')
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}\n')
    
    # Train model
    print('Starting training...\n')
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        learning_rate=LEARNING_RATE,
        save_path='best_emotion_model.pth',
        use_amp=use_amp
    )
    
    # Load best model and evaluate on test set
    print('Loading best model for testing...')
    checkpoint = torch.load('best_emotion_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('\nEvaluating on test set...')
    test_accuracy = evaluate_model(model, test_loader, device=device)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {checkpoint["val_acc"]:.2f}%')
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()

