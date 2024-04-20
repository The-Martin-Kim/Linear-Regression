import torch
import torch.optim as optim

W = torch.zeros([4, 1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

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

print(x_train.shape, y_train.shape)

optimizer = optim.SGD([W, b], lr=1e-6)

num_epochs = 6000
for epoch in range(num_epochs):

    hypothesis = x_train.matmul(W) + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, num_epochs, cost.item()
        ))

# test
test_data = torch.FloatTensor([[3.3, 700, 77, 84]])
predict = test_data.matmul(W) + b
pred_val = predict.squeeze().detach().numpy()
print('My final score is estimated as {:.2f}'.format(pred_val))
