from datasets import load_dataset
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path


def get_loader(config):
    # Vérifier si c'est un chemin local (dossier) ou un dataset Hugging Face
    dataset_path = Path(config.dataset)
    
    if dataset_path.exists() and dataset_path.is_dir():
        # Dataset local : utiliser ImageFolder
        # Structure attendue : dossier/ avec des images directement dedans
        # ou dossier/class1/, dossier/class2/, etc. (ImageFolder gère les deux)
        preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        dataset = ImageFolder(root=str(dataset_path), transform=preprocess)
        
        # Adapter pour avoir le même format que Hugging Face datasets
        class LocalDatasetWrapper:
            def __init__(self, imagefolder_dataset):
                self.dataset = imagefolder_dataset
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                image, _ = self.dataset[idx]
                return {"images": image}
        
        dataset = LocalDatasetWrapper(dataset)
        
        # Convertir en tensor batch
        def collate_fn(batch):
            images = torch.stack([item["images"] for item in batch])
            return {"images": images}
        
        loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, 
                          collate_fn=collate_fn)
    else:
        # Dataset Hugging Face
        dataset = load_dataset(config.dataset, split="train")

        preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(examples):
            images = [preprocess(image.convert("RGB")) for image in examples["image"]]
            return {"images": images}

        dataset.set_transform(transform)
        loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    return loader
