import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 2
RESULTS = 1
EPOCHS = 100
LEARNING_RATE = 0.15


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(INPUT_SIZE, 4)
        self.ln2 = nn.Linear(4, 2)
        self.ln3 = nn.Linear(2, RESULTS + 1)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(x), negative_slope=0.01)
        x = F.leaky_relu(self.ln2(x), negative_slope=0.01)
        return self.ln3(x)


def train(train_votes, train_results, test_votes, test_results):
    model = MLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    final_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        logits = model(train_votes)
        log_sm = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_sm, train_results)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_logits = model(test_votes)
            sum_ok = (test_logits.argmax(dim=-1) == test_results).float().sum().item()
            test_accuracy = sum_ok / len(test_results)
            final_accuracy = 100.0 * test_accuracy

        print(
            f"Epoch: {epoch:3} Train loss: {loss.item():8.5f} Test accuracy: {final_accuracy:5.2f}%"
        )
        if final_accuracy == 100.0:
            break

    if final_accuracy < 100.0:
        raise RuntimeError("The model is not trained well enough.")
    return model


def main():
    train_votes = torch.tensor(
        [[15, 10], [10, 15], [5, 12], [30, 20], [16, 12], [13, 25], [6, 14], [31, 21]],
        dtype=torch.float32,
    )
    train_results = torch.tensor([1, 0, 0, 1, 1, 0, 0, 1], dtype=torch.long)

    test_votes = torch.tensor(
        [[13, 9], [8, 14], [3, 10]], dtype=torch.float32
    )
    test_results = torch.tensor([1, 0, 0], dtype=torch.long)

    while True:
        print("Trying to train neural network.")
        try:
            model = train(train_votes, train_results, test_votes, test_results)
            break
        except RuntimeError as e:
            print(f"Error: {e}")
            continue

    real_world_votes = [13, 22]
    tensor_test = torch.tensor([real_world_votes], dtype=torch.float32)

    with torch.no_grad():
        result = model(tensor_test).argmax(dim=-1).item()

    print(f"real_life_votes: {real_world_votes}")
    print(f"neural_network_prediction_result: {result}")


if __name__ == "__main__":
    main()
