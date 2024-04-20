import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


x_train = torch.FloatTensor([
    [3.8, 700, 80, 50], [3.2, 650, 90, 30], [3.7, 820, 70, 40],
    [4.2, 830, 50, 70], [2.6, 550, 90, 60], [3.4, 910, 30, 40],
    [4.1, 990, 70, 20], [3.3, 870, 60, 60], [3.9, 650, 80, 50]
])

y_train = torch.FloatTensor([
    [85], [80], [78],
    [87], [85], [70],
    [81], [88], [84]
])

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-6)

num_epochs = 6000
for epoch in range(num_epochs):

    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, num_epochs, cost.item()
        ))

# test
test_data = torch.FloatTensor([[3.3, 700, 77, 84]])
predict = model(test_data)
print('My final score is estimated as {:.2f}'.format(predict.item()))
