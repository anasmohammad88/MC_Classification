import torch 
import torch.nn as nn
import torch.optim as optim
import torchmetrics as met
from torchvision import transforms
from cnn.model import MultiClassClassification
from torch.utils.tensorboard import SummaryWriter

from cnn.utility import load_data, save_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    model = MultiClassClassification().to(device)
    writer = SummaryWriter(log_dir="logs")
    
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    eval_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_data = load_data(dataset_path="./data", train=True, transform=train_transforms, batch_size=256)
    eval_data = load_data(dataset_path="./data", train=False, transform=eval_transforms, batch_size=256)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
    
    train_accuracy = met.Accuracy(task='multiclass', num_classes=10).to(device)
    eval_accuracy = met.Accuracy(task='multiclass', num_classes=10).to(device)
    best_eval_accuracy = 0.0
    
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        train_accuracy.reset()
        for image, label in train_data:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            
            pred = model(image)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy.update(pred, label)
            
        train_loss /= len(train_data)
        train_accu = train_accuracy.compute().item()
        
        
        model.eval()
        eval_loss = 0.0
        eval_accuracy.reset()
        with torch.no_grad():
            for image, label in eval_data:
                image = image.to(device)
                label = label.to(device)
                
                pred = model(image)
                loss = criterion(pred, label)
                
                eval_loss += loss.item()
                eval_accuracy.update(pred, label)
        
        eval_loss /=len(eval_data)
        eval_accu = eval_accuracy.compute().item()
        
        scheduler.step(eval_accu)
        
        writer.add_scalars("Loss", {"Train": train_loss,"Eval": eval_loss}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_accu,"Eval": eval_accu}, epoch)
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Train Accuracy: {train_accu:.4f}, Eval Accuracy: {eval_accu:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}")
        
        if eval_accu > best_eval_accuracy:
            best_eval_accuracy = eval_accu
            save_model(model, "MC-cifar-10")
            
    writer.close()
    print(f"Best Evaluation Accuracy: {best_eval_accuracy:.4f}")

            
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    train(parser.parse_args())
            
        
