from typing import List, Dict
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, set_map: List[Dict], transform=None) -> None:
        ''' Each item in set_map is expected to contain:
                img_path: Full path to image,
                label: Label corresponding to image at img_path
        '''

        self.set_map = set_map
        self.transform = transform

    def __len__(self):
        return len(self.set_map)
    
    def __getitem__(self, index):   
        sample = self.set_map[index]

        image = Image.open(sample['img_path'])

        if self.transform:
            image = self.transform(image)

        return dict(image=image, **sample)
    
class TransformWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index]['image'])
        else:
            x = self.dataset[index]['image']
        y = self.dataset[index]['label']
        return dict(image=x, label=y)
    
    def __len__(self):
        return len(self.dataset)
    
def embed(model, dataset: Dataset, batch_size=256, num_workers=4, epochs=1):
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
    embeddings = []
    labels = []
    with torch.no_grad():
        for epoch in range(epochs):
            for batch in tqdm(dataloader):
                image, label = batch['image'].cuda(), batch['label']
                embedding = model(image)
                embeddings.append(embedding)
                labels.append(label)
    embeddings = torch.cat(embeddings, dim=0).contiguous()
    labels = torch.cat(labels, dim=0).contiguous()

    return embeddings.detach().cpu().numpy(), labels.detach().cpu().numpy()

def main():
    pass

if __name__ == '__main__':
    main()