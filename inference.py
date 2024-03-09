import argparse
from model import ConvNet
import torch

def main():
    parser = argparse.ArgumentParser(description="Inference mnist data")
    parser.add_argument("--weight", type=str, default="mnist.pt",
                        help="weight file path (default: mnist.pt)")
    parser.add_argument("--image", type=str, default="data/test.jpg")
    args = parser.parse_args()

    weight_path = args.weight

    available_cuda = torch.cuda.is_available()

    device = torch.device("cuda") if available_cuda else torch.device("cpu")
    
    model_state = torch.load(weight_path)

    model = ConvNet()
    model.load_state_dict(model_state)

    x = torch.rand(1, 1, 28, 28)
    y = model(x)
    y = y.argmax(dim=1, keepdim=True)
    print("Inference result is '%d'" % y.item())


if __name__ == "__main__":
    main()