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
    Creates a DataLoader for time series data with proper preprocessing and normalization.

    Args:
        config: Configuration object containing model parameters
        device: Target device (CPU/GPU) for tensor operations
        filename: Name of the CSV file containing time series data

    Returns:
        test_loader: DataLoader object for batch processing of time series sequences
        scaler: Fitted StandardScaler object for inverse transformations
    """
    # Load data from CSV file
    df = pd.read_csv(os.path.join(config.root_path, filename))

    # Set sequence parameters from config
    pre_len = config.pre_len  # Length of prediction sequence
    train_window = config.window_size  # Size of input window

    # Reorganize dataframe to move target column to the end
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    # Extract feature columns (all columns except first)
    cols_data = df.columns[1:]
    df_data = df[cols_data]

    # Convert to numpy array
    true_data = df_data.values

    # Initialize and fit standard scaler
    scaler = StandardScaler()
    scaler.fit(true_data)

    # Normalize training data
    train_data = true_data
    train_data_normalized = scaler.transform(train_data)

    # Convert to PyTorch tensor and move to target device
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)

    # Create input-output sequences using sliding window
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)

    # Create dataset and DataLoader
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # Use config.batch_size instead of args.batch_size
        shuffle=False,  # Typically False for test/validation data
        drop_last=True  # Drop last incomplete batch
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
            prev_hidden = prev_hidden[-1]
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

##
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
    results = []
    #
    losss = []
    model = model
    model.load_state_dict(torch.load("C:\code\pythonProject9\pythonProject1\Dseqmodel2\Dseq_diur7_0.009571495.pth"))
    model.eval()
    results = []
    labels = []
    preds = []
    trues = []
    for seq, label in test_loader:
        pred = model(seq)
        mae = calculate_mae(pred.detach().cpu().numpy(),
                            np.array(label.detach().cpu()))  # MAE
        losss.append(mae)
        preds.append(pred.detach().cpu().numpy())
        trues.append(label.detach().cpu().numpy())


    preds = np.array(preds)
    trues = np.array(trues)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

    mae, mse, rmse, mape, mspe = metric(preds, trues)

    metrics_data = {
        'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE'],
        'seq2seq': [mae, mse, rmse, mape, mspe]
    }
    return metrics_data

def predict(args,df_raw,index): #index=0+96*i
    df_data = df_raw[['OT']]
    border1 = 0
    border2 = len(df_raw)-96

    scaler = StandardScaler()
    scaler.fit(df_data.values)
    data = scaler.transform(df_data.values)

    data_y=data[border1:border2]

    seq_len=96
    label_len=48
    pred_len=24
    timeenc=1
    s_begin = index
    s_end = s_begin + seq_len
    r_begin = s_end - label_len
    r_end = r_begin + label_len + pred_len
    from utils.timefeatures import time_features
    seq_x = data_y[s_begin:s_end] # 0-96
    seq_y = data_y[r_begin:r_end]
    df_stamp = df_raw[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    data_stamp = time_features(df_stamp, timeenc, 'h')
    seq_x_mark = data_stamp[s_begin:s_end]
    seq_x=torch.tensor(seq_x)
    seq_y = torch.tensor(seq_y)
    return seq_x,seq_y




def core(seq_x,seq_y,model):
    seq_len = 96
    label_len = 48
    pred_len = 24

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    bat_x = seq_x.unsqueeze(0).to(device)


    bat_x = bat_x.to(torch.float32)

    outputs = model(bat_x)
    outputs_compressed = outputs.view(-1)

    flat_outputs1 = outputs.flatten().cpu().detach()
    flat_outputs = flat_outputs1.numpy()
    outputsin = outputs_compressed.cpu().detach().unsqueeze(1)
    seq_pre = torch.cat((seq_x, outputsin), dim=0).squeeze()

    outputs_compressed = outputs.view(-1)

    true = seq_y[-24:]
    seq_true = torch.cat((seq_x, true), dim=0).squeeze()
    seq_pre = torch.cat((seq_x, outputsin), dim=0).squeeze()

    return true, flat_outputs1,  seq_true, seq_pre


def roll(args,df_raw,model):
    predsin=[]
    trues=[]
    index=0
    seq_x ,seq_y= predict(args,df_raw,index)
    true, flat_outputs,seq_true, seq_pre= core(seq_x,seq_y,model)
    for i in range(0,53):

        seq_x,seq_y=predict(args, df_raw, 24*i+index)

        true, flat_outputs, seq_true, seq_pre = core(seq_x,seq_y,model)
        predsin.append(flat_outputs)
        trues.append(true)

    return predsin,trues


if __name__ == '__main__':
    # Initialize argument parser for time series forecasting
    parser = argparse.ArgumentParser(description='Time Series forecast')

    # Model configuration parameters
    parser.add_argument('-model', type=str, default='LSTM2LSTM', help="Model type (continuously updated)")
    parser.add_argument('--other_path', type=str, default='./datatestall/*', help='Root path of the data file')
    parser.add_argument('-window_size', type=int, default=96, help="Time window size, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=24, help="Prediction length")

    # Data parameters
    parser.add_argument('-shuffle', action='store_true', default=True, help="Shuffle data in DataLoader")
    parser.add_argument('-target', type=str, default='OT', help='Target feature column to predict')
    parser.add_argument('-input_size', type=int, default=1, help='Number of features (excluding time column)')
    parser.add_argument('-feature', type=str, default='M',
                        help='[M, S, MS] - M: multivariate, S: univariate, MS: multivariate to univariate')

    # Training parameters
    parser.add_argument('-lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('-drop_out', type=float, default=0.05, help="Dropout probability for regularization")
    parser.add_argument('-epochs', type=int, default=4, help="Number of training epochs")
    parser.add_argument('-batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('-save_path', type=str, default='models', help="Path to save models")

    # Model architecture
    parser.add_argument('-hidden_size', type=int, default=128, help="Number of hidden units")
    parser.add_argument('-laryer_num', type=int, default=2, help="Number of layers")

    # Device configuration
    parser.add_argument('-use_gpu', type=bool, default=True, help="Use GPU if available")
    parser.add_argument('-device', type=int, default=0, help="GPU device index (single GPU only)")

    # Runtime options
    parser.add_argument('-train', type=bool, default=False, help="Enable training mode")
    parser.add_argument('-test', type=bool, default=True, help="Enable testing mode")
    parser.add_argument('-predict', type=bool, default=True, help="Enable prediction mode")
    parser.add_argument('-inspect_fit', type=bool, default=True, help="Enable model inspection")
    parser.add_argument('-lr-scheduler', type=bool, default=True, help="Enable learning rate scheduler")

    args = parser.parse_args()

    # Set device (GPU if available and enabled)
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Set output size based on feature mode
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # Initialize model with error handling
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>> Initializing {args.model} model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = EncoderDecoderWrapper(
            args.input_size,
            args.output_size,
            args.hidden_size,
            args.laryer_num,
            args.pre_len,
            args.window_size
        ).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>> {args.model} model initialized successfully <<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except Exception as e:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>> Failed to initialize {args.model} model <<<<<<<<<<<<<<<<<<<<<<<<<<<")
model.load_state_dict(torch.load("C:\code\pythonProject9\pythonProject1\Dseqmodel2\Dseq_diur7_0.009571495.pth"))

pathes='./datatesturegiondiur20N/*'


def region_test(args, depnum):
    """
    Perform regional testing across multiple depth files and collect results.

    Args:
        args: Configuration parameters containing model and data settings
        depnum: Number of depth files to process in the region

    Returns:
        data_dict: Dictionary containing true values and predictions for all depths
        data_dict1: Dictionary containing MSE metrics for all depths
    """
    # Initialize arrays to store true values and predictions
    num_len = 53 * 24  # 53 time windows * 24 prediction steps
    trues_np = np.zeros((depnum, num_len))
    preds_np = np.zeros((depnum, num_len))

    # Set model to evaluation mode
    model.eval()

    # Initialize containers for results
    file_names = []
    metrics_all = []
    mse_i_all = []

    # Get all data files and sort them by depth number
    file_paths = glob.glob(args.other_path, recursive=True)

    def extract_number(filename):
        """Extract depth number from filename (format: 'diurdepX.csv')"""
        return int(os.path.basename(filename).split('diurdep')[1].split('.csv')[0])

    # Process files in order of depth
    for file_path in sorted(file_paths, key=extract_number):
        file_name = os.path.basename(file_path)
        full_path = os.path.join(args.root_path, file_name)
        file_names.append(file_name)

        try:
            # Skip empty files
            if os.path.exists(full_path) and os.path.getsize(full_path) == 0:
                mse_in = 0
            else:
                # Evaluate model on current depth
                test_loader, scaler = create_dataloader3(args, device, file_name)
                metrics_data = test(model, args, test_loader, scaler)
                metrics_all.append(metrics_data)
                mse_in = metrics_data['seq2seq'][1]  # Get MSE value

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            mse_in = 0

        mse_i_all.append(mse_in)

    # Prepare metrics dictionary
    data_dict1 = {'mse_i_2d': mse_i_all}

    # Perform rolling predictions for each depth
    for i in range(1, depnum + 1):
        args.rolling_data_path = f'diurdep{i}.csv'
        file_path = os.path.join(args.root_path, args.rolling_data_path)

        try:
            df_raw = pd.read_csv(file_path)
            predsin, trues = roll(args, df_raw, model)

            # Flatten and store results
            trues_flat = torch.cat(trues, dim=0).squeeze()
            preds_flat = torch.cat(predsin, dim=0).squeeze()
            trues_np[i - 1, :] = trues_flat.numpy()
            preds_np[i - 1, :] = preds_flat.numpy()

        except Exception as e:
            print(f"Error in rolling prediction for depth {i}: {str(e)}")

    # Prepare results dictionary
    data_dict = {
        'trues': trues_np,
        'preds': preds_np
    }

    return data_dict, data_dict1
# Process all regions
all_data_dict = []
all_data_dict1 = []

# Get all regional directories
region_paths = glob.glob(pathes, recursive=True)

for region_path in tqdm(region_paths, desc="Processing regions"):
    # Configure paths for current region
    args.root_path = f"{region_path}/"
    args.other_path = f"{region_path}/*"

    # Count files in current region
    file_count = sum(len(files) for _, _, files in os.walk(region_path))

    # Test current region
    try:
        region_data, region_metrics = region_test(args, file_count)
        all_data_dict.append(region_data)
        all_data_dict1.append(region_metrics)
    except Exception as e:
        print(f"Error processing region {region_path}: {str(e)}")

# Save all results
output_folder = 'rdseq'
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, 'all_data_Dseq_u_diur_lz20N.mat')

try:
    scipy.io.savemat(save_path, {
        'data_dict': all_data_dict,
        'data_dict1': all_data_dict1
    })
    print("Processing completed successfully. Results saved to:", save_path)
except Exception as e:
    print(f"Error saving results: {str(e)}")








