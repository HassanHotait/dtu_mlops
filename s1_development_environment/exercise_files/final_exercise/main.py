import click
import torch
from model import mnistClassifier

from torch import optim, nn


from data import mnist

import matplotlib.pyplot as plt
import numpy as np

## TODO Ask How to import this function from helper (Maybe use add to path?)
def view_classify(img, ps, version="MNIST"):
    """Function for viewing an image and it's predicted classes."""
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle Boot",
            ],
            size="small",
        )
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Starting Training with lr = ",str(lr))

    # Define Model - DataLoader - Loss function - Optimizer 
    model = mnistClassifier()
    train_set, _= mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Move model to GPU if available
    model.to(device)

    epochs = 20

    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:

            # Move To Device
            images,labels =images.to(device),labels.to(device)
            
            # Train - Backpropagate - Step

            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            train_losses.append(running_loss/len(train_set))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_set)))
            

    torch.save(model.state_dict(), 'checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = mnistClassifier()
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    # Move model to GPU if available
    model.to(device)

    model.eval()

    dataiter = iter(test_set)
    images, labels = next(dataiter)
    images,labels =images.to(device),labels.to(device)
    img = images[0]
    # Convert 2D image to 1D vector
    img = img.view(1, 784)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output.cpu())

    print("PS: ",ps)
    print("PS: ",ps.shape)

    view_classify(img.cpu(),ps)
    plt.show()




cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
