from Data import Data
from Model import CNNModel


# Data Params
chars = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
seq_len = 1014
batch_size = 100

data = Data(chars, seq_len, batch_size)
inputs, labels = data.encode_dataset(True)

# Model Params
input_shape = (inputs.shape[1], inputs.shape[2])
cnn_filters = [128, 128, 128, 128, 128, 128]
cnn_kernels = [7, 7, 3, 3, 3, 3]
cnn_pools = [3, 3, None, None, None, 3]
dense_units = [256, 256]

model = CNNModel(input_shape, cnn_filters, cnn_kernels, cnn_pools, dense_units)
model.build_model()

# Training Params
epochs = 3


model.train(inputs, labels, epochs=epochs, batch_size=batch_size)
