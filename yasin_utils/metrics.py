import torch
from torch import Tensor
from typing import Any
from torchmetrics import Metric

class PerDomainAccuracy(Metric):
    def __init__(self, num_domains, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state('correct', default=torch.zeros(num_domains))
        self.add_state('total', default=torch.zeros(num_domains))
        self.num_domains = num_domains

    def update(self, preds: Tensor, targets: Tensor, domains: Tensor):
        if preds.shape != targets.shape or preds.shape != domains.shape:
            raise ValueError("preds, targets and domains must have the same shape")
        
        
        self.correct += domains[preds == targets].bincount(minlength=self.num_domains)
        self.total += domains.bincount(minlength=self.num_domains)

    def compute(self) -> torch.Tensor:
        return self.correct / self.total
    
def main():
    per_domain_accuracy = PerDomainAccuracy(num_domains=3)

    domains = torch.tensor([0,0,0,1,1,1,2,2,2])
    preds   = torch.tensor([0,1,2,0,1,2,0,1,2])
    targets = torch.tensor([0,2,0,1,1,2,0,1,2])

    per_domain_accuracy.update(preds, targets, domains)

    acc = per_domain_accuracy.compute()
    print(acc)

if __name__ == '__main__':
    main()