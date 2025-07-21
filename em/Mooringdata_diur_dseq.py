import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils.metrics import metric
import os
np.random.seed(0)


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
        #return data

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def plot_loss_data(data):
    # 使用Matplotlib绘制线图
    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
    plt.title("loss results Plot")

    plt.legend(["Loss"])

    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
def create_empty_dataset(self):
    class EmptyDataset(Dataset):
        def __init__(self):
            super(EmptyDataset, self).__init__()

        def __getitem__(self, index):
            return None

        def __len__(self):
            return 0

    return EmptyDataset()


def create_dataloader3(config, device, filename):
    """
    Creates a DataLoader for time series data with proper preprocessing
    Args:
        config: Configuration object containing parameters
        device: Device to load tensors to (CPU/GPU)
        filename: Name of the data file to load

    Returns:
        test_loader: DataLoader for the processed time series data
        scaler: Fitted StandardScaler object for inverse transformations
    """
    # Load data from CSV file
    df = pd.read_csv(os.path.join(config.root_path, filename))

    # Get prediction parameters from config
    pre_len = config.pre_len  # Length of prediction sequence
    train_window = config.window_size  # Size of input window

    # Move target column to the end
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    # Extract feature columns (all columns except first)
    cols_data = df.columns[1:]
    df_data = df[cols_data]

    # Convert to numpy array
    true_data = df_data.values

    # Initialize and fit scaler
    scaler = StandardScaler()
    scaler.fit(true_data)

    # Normalize training data
    train_data = true_data
    train_data_normalized = scaler.transform(train_data)

    # Convert to PyTorch tensor and move to device
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)

    # Create input-output sequences using sliding window
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)

    # Create dataset and DataLoader
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    return test_loader, scaler
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class LSTMEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False):
        super().__init__()
        self.seq_len=96
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.lstm2 = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.lstm3 = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)


    def forward(self, input_seq):
        seasonal_init, trend_init = self.decompsition(input_seq)

        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device='cuda')

        ct = ht.clone()
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        lstm_out, (ht, ct) = self.lstm(seasonal_init, (ht, ct))
        if self.rnn_directions > 1:
            lstm_out = lstm_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            lstm_out = torch.sum(lstm_out, axis=2)

        ht2 = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device='cuda')

        ct2 = ht2.clone()
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        lstm_out2, (ht2, ct2) = self.lstm2(trend_init, (ht2, ct2))
        if self.rnn_directions > 1:
            lstm_out2 = lstm_out2.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            lstm_out2 = torch.sum(lstm_out2, axis=2)


        return lstm_out, ht.squeeze(0),lstm_out2, ht2.squeeze(0),seasonal_init, trend_init


class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, out_put, sequence_len, hidden_size):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, input_feature_len)

    def forward(self, encoder_output, prev_hidden, y):
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]  # 保留最后一层的信息
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input), dim=-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden, rnn_hidden = self.decoder_rnn_cell(attention_combine, (prev_hidden, prev_hidden))
        output = self.out(rnn_hidden)
        return output, rnn_hidden


class EncoderDecoderWrapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, pred_len, window_size, teacher_forcing=0.3):
        super().__init__()
        self.encoder = LSTMEncoder(num_layers, input_size, window_size, hidden_size)
        self.decoder_cell = AttentionDecoderCell(input_size, output_size, window_size, hidden_size)
        self.decoder_cell2 = AttentionDecoderCell(input_size, output_size, window_size, hidden_size)
        self.output_size = output_size
        self.input_size = input_size
        self.pred_len = pred_len
        self.teacher_forcing = teacher_forcing
        self.linear = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.linear3 = nn.Linear(24, 24)
    def __call__(self, xb, yb=None):
        input_seq = xb
        encoder_output, encoder_hidden,encoder_output2, encoder_hidden2,seasonal_init, trend_init = self.encoder(input_seq)
##
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(self.pred_len, input_seq.size(0), self.input_size, device='cuda')
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size)

        y_prev = seasonal_init[:, -1, :]

        for i in range(self.pred_len):
            if (seasonal_init is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = seasonal_init[:, i]
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
            y_prev = rnn_output
            outputs[i, :, :] = rnn_output

        if self.output_size == 1:
            outputs = self.linear(outputs).permute(1, 0, 2)

        prev_hidden2 = encoder_hidden2
        if torch.cuda.is_available():
            outputs2 = torch.zeros(self.pred_len, input_seq.size(0), self.input_size, device='cuda')
        else:
            outputs2 = torch.zeros(input_seq.size(0), self.output_size)

        y_prev2 = trend_init[:, -1, :]
        for i in range(self.pred_len):
            if (trend_init is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev2 = trend_init[:, i]
            rnn_output2, prev_hidden2 = self.decoder_cell2(encoder_output2, prev_hidden2, y_prev2)
            y_prev2 = rnn_output2
            outputs2[i, :, :] = rnn_output2

        if self.output_size == 1:
            outputs2 = self.linear2(outputs2).permute(1, 0, 2)

        outputs_all=outputs2+outputs

        return outputs_all


def test(model, args, test_loader, scaler):
    """
    Evaluate model performance on test data

    Args:
        model: The trained model
        args: Configuration arguments
        test_loader: DataLoader containing test data
        scaler: Scaler object for inverse transformations
    Returns:
        metrics_data: Dictionary containing evaluation metrics
    """
    model = model
    # Load pre-trained model weights
    model.load_state_dict(torch.load("C:\code\pythonProject9\pythonProject1\Dseqmodel2\Dseq_diur7_0.009571495.pth"))
    model.eval()  # Set model to evaluation mode

    preds = []
    trues = []
    losss = []

    for seq, label in test_loader:
        pred = model(seq)
        # Calculate MAE for this batch
        mae = calculate_mae(pred.detach().cpu().numpy(),
                            np.array(label.detach().cpu()))
        losss.append(mae)
        preds.append(pred.detach().cpu().numpy())
        trues.append(label.detach().cpu().numpy())

    # Reshape predictions and true values
    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    # Calculate all metrics
    mae, mse, rmse, mape, mspe = metric(preds, trues)

    metrics_data = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE'],
        'seq2seq': [mae, mse, rmse, mape, mspe]
    }
    return metrics_data


def predict(args, df_raw, index):
    """
    Prepare input and output sequences for prediction from raw data
    """
    # Extract target column and normalize data
    df_data = df_raw[['OT']]
    border1 = 0
    border2 = len(df_raw) - 96  # Leave room for full sequence at end

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(df_data.values)
    data = scaler.transform(df_data.values)
    data_y = data[border1:border2]

    # Get sequence parameters from args
    seq_len = 96
    label_len = 48
    pred_len = 24
    timeenc = 1

    # Calculate sequence boundaries
    s_begin = index
    s_end = s_begin + seq_len
    r_begin = s_end - label_len
    r_end = r_begin + label_len + pred_len

    # Extract input and target sequences
    seq_x = data_y[s_begin:s_end]  # Input sequence (96 time steps)
    seq_y = data_y[r_begin:r_end]

    # Convert to tensors
    seq_x = torch.tensor(seq_x)
    seq_y = torch.tensor(seq_y)

    return seq_x, seq_y
def core(seq_x, seq_y, model):
    """
    Make predictions using the model and prepare results for visualization
    Returns:
        true: True values for last 24 steps
        flat_outputs1: Flattened predictions
        seq_true: Combined input and true values
        seq_pre: Combined input and predicted values
    """
    # Model configuration
    seq_len = 96
    label_len = 48
    pred_len = 24

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare input data
    bat_x = seq_x.unsqueeze(0).to(device)
    bat_x = bat_x.to(torch.float32)

    # Make prediction
    outputs = model(bat_x)
    outputs_compressed = outputs.view(-1)

    # Process outputs
    flat_outputs1 = outputs.flatten().cpu().detach()
    flat_outputs = flat_outputs1.numpy()
    outputsin = outputs_compressed.cpu().detach().unsqueeze(1)

    # Prepare sequences for visualization
    true = seq_y[-24:]
    seq_true = torch.cat((seq_x, true), dim=0).squeeze()
    seq_pre = torch.cat((seq_x, outputsin), dim=0).squeeze()

    return true, flat_outputs1, seq_true, seq_pre


def roll(args, df_raw, model):
    """
    Perform rolling window predictions using either true values or predicted values

    Args:
        args: Configuration arguments
        df_raw: Raw DataFrame containing the time series data
        model: Trained prediction model

    Returns:
        predsin: List of predictions for each rolling window
        trues: List of true values for each rolling window
    """
    predsin = []  # Store predictions
    trues = []  # Store true values
    index = 0  # Initial position in the time series

    # Get initial sequence
    seq_x, seq_y = predict(args, df_raw, index)
    true, flat_outputs, seq_true, seq_pre = core(seq_x, seq_y, model)

    # Perform rolling predictions
    for i in range(0, 10):
        # Get next sequence window (moving forward by 24 each iteration)
        seq_x, seq_y = predict(args, df_raw, 24 * i + index)

        true, flat_outputs, seq_true, seq_pre = core(seq_x, seq_y, model)

        # Store results
        predsin.append(flat_outputs)
        trues.append(true)

    return predsin, trues

if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Time Series forecast')

    # Model configuration
    parser.add_argument('-model', type=str, default='LSTM2LSTM', help="Model name")
    parser.add_argument('--other_path', type=str, default='./datatestall/*', help='Root path of the data file')
    parser.add_argument('-window_size', type=int, default=96, help="Time window size, should be > pre_len")
    parser.add_argument('-pre_len', type=int, default=24, help="Length of prediction sequence")

    # Data configuration
    parser.add_argument('-shuffle', action='store_true', default=True, help="Shuffle data in DataLoader")
   parser.add_argument('-target', type=str, default='OT', help='Target feature column')
    parser.add_argument('-input_size', type=int, default=1, help='Number of features (excluding time column)')
    parser.add_argument('-feature', type=str, default='M', help='[M, S, MS] for different prediction modes')

    # Training configuration
    parser.add_argument('-lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('-drop_out', type=float, default=0.05, help="Dropout rate to prevent overfitting")
    parser.add_argument('-epochs', type=int, default=4, help="Number of training epochs")
    parser.add_argument('-batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('-save_path', type=str, default='models', help="Path to save models")

    # Model architecture
    parser.add_argument('-hidden_size', type=int, default=128, help="Number of hidden units")
    parser.add_argument('-laryer_num', type=int, default=2, help="Number of layers")

    # Device configuration
    parser.add_argument('-use_gpu', type=bool, default=True, help="Whether to use GPU")
    parser.add_argument('-device', type=int, default=0, help="GPU device number (single GPU only)")

    # Runtime options
    parser.add_argument('-train', type=bool, default=False, help="Run training")
    parser.add_argument('-test', type=bool, default=True, help="Run testing")
    parser.add_argument('-predict', type=bool, default=True, help="Run prediction")
    parser.add_argument('-inspect_fit', type=bool, default=True, help="Inspect model fit")
    parser.add_argument('-lr-scheduler', type=bool, default=True, help="Use learning rate scheduler")


    args = parser.parse_args()

    # Set device
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Set output size based on feature type
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # Initialize model
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>Initializing {args.model} model<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = EncoderDecoderWrapper(args.input_size, args.output_size, args.hidden_size, args.laryer_num,
                                      args.pre_len, args.window_size).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>{args.model} model initialized successfully<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>Failed to initialize {args.model} model<<<<<<<<<<<<<<<<<<<<<<<<<<<")
args.root_path = './diurmooring_data/'
args.other_path = './diurmooring_data/*'

num_files = 40
num_len=10*24
trues_np = np.zeros((num_files, num_len))
preds_np = np.zeros((num_files, num_len))
model.load_state_dict(torch.load("C:\code\pythonProject9\pythonProject1\Dseqmodel2\Dseq_diur7_0.009571495.pth"))
# Set model to evaluation mode
model.eval()

# Initialize lists to store results
file_names = []
metrics_all = []
mse_i_all = []

# Get sorted list of data files
file_paths = glob.glob(args.other_path, recursive=True)


def extract_number(filename):
    """Extract numeric part from filename (format: 'diurdepX.csv')"""
    return int(filename.split('diurdep')[1].split('.csv')[0])


# Sort files by their numeric part
sorted_file_paths = sorted(file_paths, key=extract_number)

# Process each file with progress bar
for file_path in tqdm(sorted_file_paths):
    file_name = os.path.basename(file_path)
    file_names.append(file_name)
    full_path = os.path.join(args.root_path, file_name)

    try:
        # Skip empty files
        if os.path.exists(full_path) and os.path.getsize(full_path) == 0:
            mse_in = 0
        else:
            # Load data and evaluate model
            test_loader, scaler = create_dataloader3(args, device, file_name)
            metrics_data = test(model, args, test_loader, scaler)

            metrics_all.append(metrics_data)
            mse_in = metrics_data['seq2seq'][1]  # Get MSE value

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        mse_in = 0

    mse_i_all.append(mse_in)

# Prepare results for saving
mat_dict = {
    'mse_i_2d': mse_i_all,  # 2D array of MSE values
    # 'mse_h_2d': mse_h_2d  # Uncomment if needed
}

# Save results to .mat file (commented out - uncomment to use)
# scipy.io.savemat('results.mat', mat_dict)


import scipy.io as sio
from scipy.io import savemat
# Save both arrays in a single .mat file
folder = 'rdseq'
save_path = os.path.join(folder, 'mse_mooringdata_diur.mat')
savemat(save_path, mat_dict)

for i in range(1,40):
    args.rolling_data_path = f'diurdep{i}.csv'
    file_path = os.path.join(args.root_path, args.rolling_data_path)
    df_raw = pd.read_csv(file_path)
    predsin, trues = roll(args, df_raw,model)
    trues_flat = torch.cat(trues, dim=0).squeeze()
    preds_flat = torch.cat(predsin, dim=0).squeeze()
    trues_np[i-1,:] = trues_flat.numpy()
    preds_np[i-1,:] = preds_flat.numpy()
from scipy.io import savemat
data_dict = {
    'trues': trues_np,
    'preds': preds_np,
}
folder = 'rdseq'
save_path = os.path.join(folder, 'mooringdata_diur.mat')
savemat(save_path, data_dict)












