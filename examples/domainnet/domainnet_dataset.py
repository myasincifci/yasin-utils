import os
from yasin_utils.data import ImageDataset
from domainnet_metadata import DOMAIN_NET_CLASSES, DOMAIN_NET_DOMAINS 

class DomainNetDataset(ImageDataset):
    def __init__(self, root: str, transform=None, domains=None) -> None:
        self.domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'] if not domains else domains
        set_map = []
        for domain in self.domains:
            try:
                labels = os.listdir(os.path.join(root, domain))
            except:
                raise Exception(f'{domain} directory not found.')
            for label in labels:
                for image in os.listdir(os.path.join(root, domain, label)):
                    set_map.append(
                        dict(
                            img_path=os.path.join(root, domain, label, image),
                            label=label,
                            domain=domain
                            )
                        )

        super().__init__(set_map, transform)

        self.num_classes = len(DOMAIN_NET_CLASSES)
        self.num_domains = len(DOMAIN_NET_DOMAINS)
        self.class_map = dict(zip(DOMAIN_NET_CLASSES, range(len(DOMAIN_NET_CLASSES))))
        self.domain_map = dict(zip(DOMAIN_NET_DOMAINS, range(len(DOMAIN_NET_DOMAINS))))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['label'] = self.class_map[item['label']]
        item['domain'] = self.domain_map[item['domain']]

        return item

def main():
    dataset = DomainNetDataset(root='/data/domainnet')
    print(dataset.__getitem__(0))

if __name__ == '__main__':
    main()