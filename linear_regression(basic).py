import torch
import torch.optim as optim

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

x_train = torch.FloatTensor([
    [78], [83], [56], [67], [85], [44], [32], [90]
])

y_train = torch.FloatTensor([
    [66], [73], [76], [65], [81], [54], [29], [85]
])

print(x_train.shape, y_train.shape)

optimizer = optim.SGD([W, b], lr=0.0001)

num_epochs = 100
for epoch in range(num_epochs):

    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, num_epochs, cost.item()
        ))

# test
predict = W * 71 + b
print('My final score is estimated as {:.2f}'.format(predict.item()))
