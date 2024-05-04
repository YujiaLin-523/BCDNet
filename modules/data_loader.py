import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# Define the transformation
transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                     transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(30), transforms.Resize((256, 256))])


# Define the dataset class
class BCDDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transforms = transform
        self.imgs = []
        self.label_to_index = {"0": 0, "1": 1}  # '0' for negative, '1' for positive

        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    img_path = os.path.join(dirpath, filename)
                    label = dirpath.split('/')[-1]
                    self.imgs.append((img_path, label))

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        image = Image.open(img_path)
        if self.transforms is not None:
            image = self.transforms(image)
        # Convert label to numerical value
        label = self.label_to_index[label]
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)


# Split the dataset into training, validation and testing sets
def split(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Display the number of samples in each set
    print('Number of samples in dataset: ', len(dataset))
    print('Number of samples in training set: ', len(train_set))
    print('Number of samples in validation set: ', len(val_set))
    print('Number of samples in testing set: ', len(test_set))

    return train_set, val_set, test_set


# Create the TrainLoader, ValLoader and TestLoader
class TrainLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=16):
        super(TrainLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class ValLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=16):
        super(ValLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class TestLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=16):
        super(TestLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
