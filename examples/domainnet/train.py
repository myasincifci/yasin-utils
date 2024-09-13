import PIL.Image
import jax
import numpy as np
import jax.numpy as jnp

import PIL
from torch.utils.data import DataLoader, random_split
from domainnet_dataset import DomainNetDataset

from jax_trainer import TrainerModule
from flax import linen as nn
from flaxmodels import ResNet18

def transform(img: PIL.Image):
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    return img

def train_classifier(*args, num_epochs=200, train_loader, val_loader, test_loader, **kwargs):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(*args, **kwargs)
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(train_loader, val_loader, num_epochs=num_epochs)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
    # Test trained model
    val_acc = trainer.eval_model(val_loader)
    test_acc = trainer.eval_model(test_loader)
    return trainer, {'val': val_acc, 'test': test_acc}

def numpy_collate(batch):
    images = []
    labels = []
    domains = []

    for d in  batch:
        for k, v in d.items():
            match k:
                case 'image':
                    images.append(v)
                case 'label':
                    labels.append(v)
                case _:
                    pass

    return np.stack(images), np.stack(labels)

def main():
    dataset = DomainNetDataset(root='/data/domainnet', transform=transform)
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_set, 16, shuffle=True, collate_fn=numpy_collate)
    val_loader = DataLoader(val_set, 16, shuffle=False, collate_fn=numpy_collate)
    test_loader = DataLoader(test_set, 16, shuffle=False, collate_fn=numpy_collate)

    resnet_trainer, resnet_results = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_class=ResNet18,
        model_name='ResNet18',
        model_hparams={
            'num_classes': dataset.num_classes,
            'pretrained': None        
        },
        optimizer_name="adamw",
        optimizer_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4
        },
        exmp_imgs=jax.device_put(next(iter(train_loader))[0]),
        num_epochs=200
    )

    print(resnet_results)

    a=1

if __name__ == '__main__':
    main()