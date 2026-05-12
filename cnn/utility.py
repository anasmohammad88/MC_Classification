import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

CLASS_TO_IDX = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}

class ImageDataset(Dataset):
    def __init__(self, dataset_path, train=True, transform=None):
        self.dataset_path = Path(dataset_path)
        self.train = train
        self.transform = transform
        self.download_dataset()
        self.samples = self.read_samples()
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image , label = self.samples[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def read_samples(self):
        samples = [] 
        
        cifar_dataset = datasets.CIFAR10(root=self.dataset_path, train=self.train, download=False)
        
        for image, label in cifar_dataset:
            samples.append((image, label))

        return samples
        
        
    def download_dataset(self):
        cifar_folder = self.dataset_path / "cifar-10-batches-py"
        
        if not cifar_folder.exists():
            datasets.CIFAR10(root=self.dataset_path, train=True, download=True)
            datasets.CIFAR10(root=self.dataset_path, train=False, download=True)
            print("Download completed.")
            
    
def load_data(dataset_path, train=True, transform=None, batch_size=128, num_workers=2):
    image_dataset = ImageDataset(dataset_path = dataset_path, train=train, transform=transform)
    return DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

def show_sample(data_loader):
    images, labels = next(iter(data_loader))
    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i + 1)

        # Resize tensor image to 128x128
        image = resize(images[i], [128, 128])
        image = image.permute(1, 2, 0)
        plt.imshow(image, interpolation='nearest')
        class_name = IDX_TO_CLASS[labels[i].item()]
        plt.title(class_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    
    
def save_model(model, name="model"):
    path = Path("checkpoints") / f"{name}.pt"
    path.parent.mkdir(exist_ok=True)

    torch.save(model.state_dict(), path)
    return path


def load_model(model, name="model", device="cpu"):
    path = Path("checkpoints") / f"{name}.pt"

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_path = "./data"
        
        train_loader = load_data(dataset_path=dataset_path, train=True, transform=transform)
        test_loader = load_data( dataset_path=dataset_path, train=False, transform=transform)
        show_sample(train_loader)

            
        