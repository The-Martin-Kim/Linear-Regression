import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


x_train = torch.FloatTensor([
    [78], [83], [56], [67], [85], [44], [32], [90]
])

y_train = torch.FloatTensor([
    [66], [73], [76], [65], [81], [54], [29], [85]
])

model = LinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 100
for epoch in range(num_epochs):

    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, num_epochs, cost.item()
        ))

# test
test_data = torch.FloatTensor([[71]])
predict = model(test_data)
print('My final score is estimated as {:.2f}'.format(predict.item()))
