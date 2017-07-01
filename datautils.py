import torchvision.transforms as transforms

def getTransform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def count(iter):
    return sum(1 for _ in iter)

