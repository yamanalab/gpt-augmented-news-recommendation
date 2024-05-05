import torch


batch_size = 10
max_length = 30
emb_dim = 16

window_size = 3
num_filters = 300

cnn = torch.nn.Conv1d(emb_dim, num_filters, kernel_size=(3), padding="same")

print("Weight & Bias: ", cnn.weight.size(), cnn.bias.size())


input = torch.randn((batch_size, max_length, emb_dim))
input = torch.transpose(1, 2)
# input = torch.unsqueeze(input, 1)

print("Input: ", input.size())

output = cnn(input)

print("Output: ", output.size())
