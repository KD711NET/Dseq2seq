# The code framework is adapted from [https://blog.csdn.net/java1314777/article/details/134864319].
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
        # return data

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean



def plot_loss_data(data):

    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')

    # 添加标题
    plt.title("loss results Plot")

    # 显示图例
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
def create_empty_dataset():
    class EmptyDataset(Dataset):
        def __init__(self):
            super(EmptyDataset, self).__init__()

        def __getitem__(self, index):
            return None

        def __len__(self):
            return 0

    return EmptyDataset()
empty_dataset = create_empty_dataset()
def createtest_loader2(config,test_path,root2_path,device,empty_dataset):
    file_names = []
    file_paths = glob.glob(test_path, recursive=True)
    for file_path in file_paths:
        # 获取文件名部分并添加到列表中
        file_name = os.path.basename(file_path)
        file_names.append(file_name)

    concat_dataset = empty_dataset
    for i in file_names:
        try:
            df = pd.read_csv(os.path.join(root2_path,
                                          i))
            df_data = df[['OT']]
            if df_data.isnull().values.any():
                print(f"File {file_path} contains NaN values. Skipping.")
                continue

            if not df.empty:
                pre_len = config.pre_len
                train_window = config.window_size

                # 将特征列移到末尾
                target_data = df[[config.target]]
                df = df.drop(config.target, axis=1)
                df = pd.concat((df, target_data), axis=1)

                cols_data = df.columns[1:]
                df_data = df[cols_data]

                true_data = df_data.values

                scaler = StandardScaler()
                scaler.fit(true_data)

                train_data = true_data

                train_data_normalized = scaler.transform(train_data)

                train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)

                train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)

                train_dataset = TimeSeriesDataset(train_inout_seq)

                concat_dataset = ConcatDataset([concat_dataset, train_dataset])


            else:

                continue
        except pd.errors.EmptyDataError:
            continue

    test_loader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


    return test_loader, scaler


#trainloader加载
def create_dataloader2(config, device,empty_dataset):

    file_names = []
    file_paths = glob.glob(config.other_path, recursive=True)
    for file_path in file_paths:
        # 获取文件名部分并添加到列表中
        file_name = os.path.basename(file_path)
        file_names.append(file_name)

    # empty_dataset = create_empty_dataset()
    concat_dataset=empty_dataset
    for i in file_names:
        try:
            df = pd.read_csv(os.path.join(config.root_path,
                                          i))
            df_data=df[['OT']]
            if df_data.isnull().values.any():
                print(f"File {file_path} contains NaN values. Skipping.")
                continue

            if not df.empty:
                pre_len = config.pre_len
                train_window = config.window_size


                target_data = df[[config.target]]
                df = df.drop(config.target, axis=1)
                df = pd.concat((df, target_data), axis=1)

                cols_data = df.columns[1:]
                df_data = df[cols_data]


                true_data = df_data.values


                scaler = StandardScaler()
                scaler.fit(true_data)

                train_data = true_data

                train_data_normalized = scaler.transform(train_data)


                train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)


                train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)

                # 创建数据集
                train_dataset = TimeSeriesDataset(train_inout_seq)

                concat_dataset = ConcatDataset([concat_dataset, train_dataset])


            else:

                continue

        except pd.errors.EmptyDataError:
            continue



    train_loader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


    return train_loader, scaler


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


def train(model, args, scaler, device):
    # Record start time for training duration
    start_time = time.time()

    # Initialize model, loss function and optimizer
    model = model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = args.epochs

    # Initialize lists to track losses
    results_loss = []
    losses = []

    # Training loop
    for i in tqdm(range(epochs)):
        train_loss = []
        model.train()

        # Training phase
        for seq, labels in tqdm(train_loader):
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            train_loss.append(single_loss.detach().cpu().numpy())

        # Store average training loss for this epoch
        results_loss.append(np.average(train_loss))

        # Create directory for saving models
        folder_name = 'Dseqmodel2'
        os.makedirs(folder_name, exist_ok=True)

        # Validation phase
        valid_loss = []
        model.eval()
        for seq, labels in test_loader:
            pred = model(seq)
            mse = F.mse_loss(pred.detach().cpu(), labels.detach().cpu())
            valid_loss.append(mse)

        # Calculate and store average validation loss
        valid_loss2 = np.average(valid_loss)
        losses.append(valid_loss2)

        # Prepare data for saving
        ee1 = str(valid_loss2)
        data_dict = {
            'results_loss': results_loss,
            'losses': losses
        }
        ee2 = str(i)

        # Save model checkpoint
        file_path = os.path.join(folder_name, f'Dseq_diur{ee2}_{ee1}.pth')
        torch.save(model.state_dict(), file_path)

        # Save loss values to .mat file
        file_name = 'Dseqmodel_loss\loss2_values' + ee2 + '.mat'
        scipy.io.savemat(file_name, data_dict)
        print(f"Saved results_loss and losses to {file_name}")

    # Print training summary
    training_duration = (time.time() - start_time) / 60
    print(f">>>>>>>>>>>>>>>>>>>>>>Model saved, training time: {training_duration:.4f} min<<<<<<<<<<<<<<<<<<")

    # Plot and save loss data
    plot_loss_data(results_loss)
    np.save('seq2seq=1.npy', results_loss)


if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Time Series forecast')

    # Model configuration arguments
    parser.add_argument('-model', type=str, default='LSTM2LSTM')
    parser.add_argument('--other_path', type=str, default='./diurtrain/*')
    parser.add_argument('-window_size', type=int, default=96)
    parser.add_argument('-pre_len', type=int, default=24)
    parser.add_argument('--root_path', type=str, default='./diurtrain/')
    parser.add_argument('-shuffle', action='store_true', default=True)

    parser.add_argument('-target', type=str, default='OT')
    parser.add_argument('-input_size', type=int, default=1)
    parser.add_argument('-feature', type=str, default='M')
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-drop_out', type=float, default=0.05)
    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-save_path', type=str)

    # Model architecture arguments
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-laryer_num', type=int, default=2)

    # Device configuration
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-device', type=int, default=0)

    # Runtime options
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-predict', type=bool, default=True)
    parser.add_argument('-inspect_fit', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)


    args = parser.parse_args()

    # Set device (GPU/CPU)
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")

    # Create data loaders
    train_loader, scaler = create_dataloader2(args, device, empty_dataset)
    test_path = './diurvalid/*'
    root2_path = './diurvalid/'
    test_loader, scaler2 = createtest_loader2(args, test_path, root2_path, device, empty_dataset)

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

    # Train model if specified
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>Starting {args.model} model training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device)
