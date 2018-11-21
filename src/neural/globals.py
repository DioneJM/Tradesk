# Hyper-parameters stored here

# Size of the training data. This is approximately 90% of the total data provided
train = 8000

# Window size for batch normalization
window = 2000

# Exponential moving average parameter for input smoothing
gamma = 0.1

# Dimensionality of the data. This is a simple one-dimensional time series data
dimension = 1

# Size of the sequence given to the memory layer
sequence_size = 50

# Number of samples per element of the sequence
batch_size = 500

# Number of hidden nodes per layer of the deep memory network we are using
num_nodes = [200, 200, 150]

# The depth of the memory network
number_layers = len(num_nodes)

# Prevent model over-fitting
dropout = 0.2

# Clipping value to circumvent exploding gradients
clip = 5.0

# Number of epochs
epochs = 1

# Sample of points to evaluate model performance
testing_range = 1000

# Number of values to predict in the future
number_of_future = 6

# Directory name to store the best performing model
model_directory_path = 'Model'

# File name of csv containing predictions
csv_file_name = '/home/bennydai/PythonThingo/output/predictions.csv'

index_directory = '/home/bennydai/Applications/predictions.csv'

# Remote Computer Directory
put_address = '/Users/lindsaypage/Desktop/predictions.csv'

# SSH details
server = '172.20.10.9'

port = 22

username = 'lindsaypage'

password = 'l16220913'

