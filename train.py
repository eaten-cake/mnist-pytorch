import argparse
from torchvision import datasets, transforms
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from model import ConvNet

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # Print once per batch
        print("Train Epoch: %d [%d/%d (%.0f)]\tLoss: %.6f" 
              % 
              (epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))    

def main():
    parser = argparse.ArgumentParser(description = "train mnist dataset setting")
    parser.add_argument("--batch-size", type=int, default=4, metavar="N",
                        help = "input batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR",
                        help="learning rate (default: 1.0)")
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    available_cuda = torch.cuda.is_available()

    device = torch.device("cuda") if available_cuda else torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}

    if available_cuda:
        cuda_kwargs = {"num_workers": 1,
                    "pin_memory": True,
                    "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST("./datasets", train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST("./datasets", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ConvNet().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print(len(train_loader))

    # for epoch in range(0, args.epochs):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist.pt")
    


if __name__ == "__main__":
    main()