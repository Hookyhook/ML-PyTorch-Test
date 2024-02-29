import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

showCurrentProgress = True

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 16),
            nn.Tanh(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)

# Lists to store accuracy and last guesses
accuracy_list = []
last_guesses = []
last_wrong_guesses = []


def initializeGraph():
    fig = plt.figure(layout="constrained", figsize=(15, 7))

    gs = GridSpec(2, 5, figure=fig)

    ax_accuracy = fig.add_subplot(gs[0, :-1])
    ax_last_guess = fig.add_subplot(gs[0, -1:])

    ax_last_wrong_guesses = []

    for i in range(0, 5):
        ax_last_wrong_guesses.append(fig.add_subplot(gs[1, i]))

    fig.suptitle("Mnist Dataset")

    return ax_accuracy, ax_last_guess, ax_last_wrong_guesses


ax_accuracy, ax_last_guess, ax_last_wrong_guesses = initializeGraph();


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    total = 0
    global accuracy_list  # Access global variable inside the function
    global last_guesses
    global last_wrong_guesses
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        if batch % 100 == 0:
            for i in range(len(predicted)):
                last_guesses.append((X[i], predicted[i], y[i]))
                if predicted[i] != y[i]:
                    last_wrong_guesses.append((X[i], predicted[i], y[i]))
                    if len(last_wrong_guesses) > 5:
                        last_wrong_guesses.pop(0)
                if len(last_guesses) > 5:
                    last_guesses.pop(0)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            accuracy = correct / total * 100
            accuracy_list.append(accuracy)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] Accuracy: {accuracy:.2f}%")
            # Plotting accuracy during training
            ax_accuracy.clear()  # Clear previous plot
            ax_accuracy.plot(accuracy_list, '-o')
            ax_accuracy.set_title('Accuracy during Training')
            ax_accuracy.set_ylabel('Accuracy (%)')
            ax_accuracy.grid(True)
            if showCurrentProgress:
                # Display last guesses on the plot
                for i, (image, predicted, target) in enumerate(last_guesses, 1):
                    ax_last_guess.imshow(image.squeeze().cpu().numpy(), cmap='gray')

                    if predicted == target:
                        title = '(Correct)'
                    else:
                        title = f'Prediction: {predicted}, Target: {target} (Wrong)'
                    ax_last_guess.set_title(title)
                    ax_last_guess.axis('off')

                # Display last correct guesses on the plot
                for i, (image, predicted, target) in enumerate(last_wrong_guesses, 1):
                    ax = ax_last_wrong_guesses[i-1]
                    image = image.squeeze().cpu().numpy()  # Assuming image is a torch.Tensor
                    ax.imshow(image, cmap='gray')
                    ax.set_title(f'P:{predicted} T:{target}')
        plt.pause(0.0001)  # Pause to update plot


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

plt.show()
