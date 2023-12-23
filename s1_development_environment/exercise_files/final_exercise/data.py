import torch
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CorruptMNISTDataset(Dataset):
    """Corrupt MNIST dataset."""

    def __init__(self, dirPath,train = True ,transform=None):
        """
        Arguments:
            dirPath (string): Path to dir where Pytorch files are stored.
            train (boolean): Training Set or Test Set
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Load Training Images From 6 Sets
        if train == True:
            for i in range(6):

                if i == 0:
                    self.images = torch.load(os.path.join(dirPath,"train_images_" + str(i) + ".pt"))
                    self.target = torch.load(os.path.join(dirPath,"train_target_" + str(i) + ".pt"))

                else:
                    trainImgsFile = torch.load(os.path.join(dirPath,"train_images_" + str(i) + ".pt"))
                    trainTargetFile = torch.load(os.path.join(dirPath,"train_target_" + str(i) + ".pt"))

                    self.images = torch.cat((self.images,trainImgsFile),dim = 0)
                    self.target = torch.cat((self.target,trainTargetFile),dim = 0)

        else:
            self.images = torch.load(os.path.join(dirPath,"test_images.pt"))
            self.target = torch.load(os.path.join(dirPath,"test_target.pt"))



        if transform != None:
            ## TODO Check ISSUE
            ToPIL = transforms.ToPILImage()
            self.images = transform(ToPIL(self.images))

    def __len__(self):
        return len(self.images)
        

    def __getitem__(self, idx):
        return self.images[idx], self.target[idx]


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset

    dataPath = "C:/Users/Hasan/OneDrive/Desktop/Projects/dtu_mlops/data/corruptmnist"

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                ])
    
    trainSet = CorruptMNISTDataset(dirPath = dataPath, train = True, transform=None)
    testSet = CorruptMNISTDataset(dirPath = dataPath, train = False, transform=None)

    print("Length of TrainSet: ", len(trainSet))
    print("Length of TestSet: ", len(testSet))

    train = DataLoader(trainSet, batch_size = 64, shuffle = True)
    test = DataLoader(testSet, batch_size = 64, shuffle = True)

    
    return train, test




