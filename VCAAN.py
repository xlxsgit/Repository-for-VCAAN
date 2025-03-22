# libraries
import time
import random
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from patsy import dmatrix
from functools import partial
from fancyimpute import KNN, IterativeImputer

import tsdb
from pypots.imputation import  LOCF, GRUD, USGAN, SAITS, iTransformer
from pypots.utils.metrics import calc_mse
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, mnar_x, mnar_t, seq_missing, block_missing

import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset

# ignore warnings, set device and start time
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {torch.cuda.get_device_name(0)}')

start_time = time.time()

# functions and classes

# function: set random seed for reproducibility
def set_my_seed(seed=15):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Guaranteed reproducibility of convolution operations
    torch.backends.cudnn.benchmark = False

# function: load data
def run_load_data():  # 加载数据
    data = tsdb.load("beijing_multisite_air_quality", use_cache=True)['X']
    datas = data.loc[:, ['No', 'PM2.5','station']].copy()
    datas = datas[(start_day*24<datas.No) & (datas.No<=start_day*24+use_days*24)] # load data from start_day to start_day+use_days, each day has 24 hours
    datas['station'] = datas['station'].astype('category').cat.codes
    Y = datas.pivot(index='No', columns='station', values='PM2.5').reset_index(drop=True).to_numpy() # target data
    Y = StandardScaler().fit_transform(Y) # standardize

    num_features = Y.shape[-1]  # The number of sites
    print(f'Number of sites: {num_features}')
    n_steps = 24  # The time step of one sample
    num_samples = Y.shape[0] // n_steps  # The number of samples
    Y = Y.reshape(num_samples, n_steps, num_features)  # Convert to the shape required by the library

    Y_ori = Y.copy()  # Backup original data
    Y = torch.tensor(Y, dtype=torch.float32, device=device)  # Convert to tensors
    Y_ori = torch.tensor(Y_ori, dtype=torch.float32, device=device)  # Convert to tensors
    return Y, Y_ori, num_features, n_steps, num_samples # Return data, original data, number of sites, time steps, number of samples

# function: make missing
def run_make_miss(Y, miss_type, miss_rate):  # miss_type: 1~5, miss_rate: 1~4
    set_my_seed()
    dict_type_rate = {1: [0.105, 0.3, 0.5],
                      2: [0.125, 0.48, 3.1],
                      3: [1.42, 0.595, -0.16],
                      4: [0.1, 0.3, 0.5],
                      5: [0.0175, 0.0558, 0.11]}  # Parameters corresponding to the deletion rate of different deletion types
    target_rate = dict_type_rate[miss_type][miss_rate - 1]  # target missing rate
    Y = Y.to('cpu')
    if miss_type == 1:
        Y = mcar(Y, p=target_rate)
    elif miss_type == 2:
        Y = mnar_t(Y, scale=target_rate, cycle=24, pos=6)
    elif miss_type == 3:
        Y = mnar_x(Y, offset=target_rate)
    elif miss_type == 4:
        Y = seq_missing(Y, p=target_rate, seq_len=12)
    elif miss_type == 5:
        Y = block_missing(Y, factor=target_rate, block_len=7, block_width=7)
    Y = Y.to(device)
    return Y

# function: split data and indicator
def run_split_data_and_indicator(data, data_ori, num_features, num_samples):
    # Split the data into training, validation, and test sets
    num_train = int(num_samples * 0.5)
    num_val = int(num_samples * 0.3)

    # Dataset in tensor form
    t_data_train, t_data_val, t_data_test = (data[:num_train],
                                             data[num_train:num_train + num_val],
                                             data[num_train + num_val:])
    t_data_ori_train, t_data_ori_val, t_data_ori_test = (data_ori[:num_train],
                                                         data_ori[num_train:num_train + num_val],
                                                         data_ori[num_train + num_val:])
    # Dataset in matrix form
    m_data_train, m_data_val, m_data_test = (t_data_train.reshape(-1, num_features).clone(),
                                             t_data_val.reshape(-1, num_features).clone(),
                                             t_data_test.reshape(-1, num_features).clone())
    m_data_ori_train, m_data_ori_val, m_data_ori_test = (t_data_ori_train.reshape(-1, num_features).clone(),
                                                         t_data_ori_val.reshape(-1, num_features).clone(),
                                                         t_data_ori_test.reshape(-1, num_features).clone())

    # The form of data required for POTS
    I_train = {"X": t_data_train}
    I_val = {"X": t_data_val, "X_ori": t_data_ori_val}
    I_test = {"X": t_data_test}

    # indicator for tensor form
    t_indicator_train = torch.isnan(t_data_train) ^ torch.isnan(t_data_ori_train)
    t_indicator_val = torch.isnan(t_data_val) ^ torch.isnan(t_data_ori_val)
    t_indicator_test = torch.isnan(t_data_test) ^ torch.isnan(t_data_ori_test)

    # indicator for matrix form
    m_indicator_train, m_indicator_val, m_indicator_test = (t_indicator_train.reshape(-1, num_features).clone(),
                                                            t_indicator_val.reshape(-1, num_features).clone(),
                                                            t_indicator_test.reshape(-1,
                                                                                     num_features).clone())  # Note that the lag part is not removed here

    return (I_train, I_val, I_test,
            t_data_train, t_data_val, t_data_test,
            t_data_ori_train, t_data_ori_val, t_data_ori_test,
            t_indicator_train, t_indicator_val, t_indicator_test,
            m_data_train, m_data_val, m_data_test,
            m_data_ori_train, m_data_ori_val, m_data_ori_test,
            m_indicator_train, m_indicator_val, m_indicator_test)

# function: baselines
def baselines(model_name, t_data_train, t_data_val, t_data_train_ori, t_data_val_ori, t_indicator_train, m_data_train):
    t_data_train = t_data_train.cpu().detach().numpy()
    t_data_val = t_data_val.cpu().detach().numpy()
    t_data_train_ori = t_data_train_ori.cpu().detach().numpy()
    t_data_val_ori = t_data_val_ori.cpu().detach().numpy()
    t_indicator_train = t_indicator_train.cpu().detach().numpy()
    m_data_train = m_data_train.cpu().detach().numpy()
    if model_name in ['MICE', 'KNN']:
        if model_name == 'MICE':
            t_hat_train = IterativeImputer().fit_transform(m_data_train).reshape(-1, n_steps, num_features)
        elif model_name == 'KNN':
            t_hat_train = KNN(verbose=False).fit_transform(m_data_train).reshape(-1, n_steps, num_features)
        loss_train = calc_mse(t_hat_train, np.nan_to_num(t_data_train_ori), t_indicator_train)
        return torch.tensor(t_hat_train, dtype=torch.float32, device=device), loss_train
    elif model_name in ['LOCF', 'GRUD', 'USGAN', 'SAITS', 'iTransformer']:
        I_train = {"X": t_data_train}
        I_val = {"X": t_data_val, "X_ori": t_data_val_ori}
        if model_name == 'GRUD':
            model = GRUD(n_steps=n_steps, n_features=num_features, rnn_hidden_size=64, epochs=epochs_BASE,
                         verbose=False)
            model.fit(I_train, I_val)
        elif model_name == 'SAITS':
            model = SAITS(n_steps=n_steps, n_features=num_features, n_layers=4, d_model=32, n_heads=4, d_k=8,
                          d_v=32, d_ffn=64, epochs=epochs_BASE, verbose=False)
            model.fit(I_train, I_val)
        elif model_name == 'iTransformer':
            model = iTransformer(n_steps=n_steps, n_features=num_features, n_layers=4, d_model=32, n_heads=4, d_k=8,
                                 d_v=32, d_ffn=64, epochs=epochs_BASE, verbose=False)
            model.fit(I_train, I_val)
        elif model_name == 'USGAN':
            model = USGAN(n_steps=n_steps, n_features=num_features, rnn_hidden_size=64, epochs=epochs_BASE, verbose=False)
            model.fit(I_train, I_val)
        elif model_name == 'LOCF':
            model = LOCF()
        t_hat_train = model.impute(I_train)
        loss_train = calc_mse(t_hat_train, np.nan_to_num(t_data_train_ori), t_indicator_train)
        return torch.tensor(t_hat_train, dtype=torch.float32, device=device), loss_train
    else:
        raise ValueError('model_name must be one of MICE, KNN, LOCF, GRUD, USGAN, SAITS, iTransformer')

# function: run baselines
def run_baselines(t_data_train, t_data_val, t_data_train_ori, t_data_val_ori, t_indicator_train, m_data_train):
    partial_baselines = partial(baselines,
                                t_data_train=t_data_train, t_data_val=t_data_val,
                                t_data_train_ori=t_data_train_ori, t_data_val_ori=t_data_val_ori,
                                t_indicator_train=t_indicator_train, m_data_train=m_data_train)
    t_hat_train_mice, loss_train_mice = partial_baselines('MICE')
    t_hat_train_knn, loss_train_knn = partial_baselines('KNN')
    t_hat_train_locf, loss_train_locf = partial_baselines('LOCF')
    t_hat_train_grud, loss_train_grud = partial_baselines('GRUD')
    t_hat_train_usgan, loss_train_usgan = partial_baselines('USGAN')
    t_hat_train_saits, loss_train_saits = partial_baselines('SAITS')
    t_hat_train_itransformer, loss_train_itransformer = partial_baselines('iTransformer')

    return (t_hat_train_mice, loss_train_mice,
            t_hat_train_knn, loss_train_knn,
            t_hat_train_locf, loss_train_locf,
            t_hat_train_grud, loss_train_grud,
            t_hat_train_usgan, loss_train_usgan,
            t_hat_train_saits, loss_train_saits,
            t_hat_train_itransformer, loss_train_itransformer)

# function: get CONS and VARY data, corresponding to CCA and VCA, respectively
def get_cons_vary(t_hat, m_indicator, select):  # Generate training and test data
    m_hat = t_hat.reshape(-1, num_features)
    T, N = m_hat.shape

    ts_R = m_hat.clone()
    ts_M = m_indicator.clone()
    ts_W = tensor(np.corrcoef(m_hat.cpu().detach().numpy(), rowvar=False) - np.eye(N), dtype=torch.float32,
                  device=device)

    ts_X_1 = ts_R.clone()[lag - 1:(T - 2 * lag) + lag - 1, :]
    ts_X_2 = ts_R.clone()[lag + 1:(T - 2 * lag) + lag - 1 + lag + 1, :]
    ts_X_3 = torch.mm(ts_R.clone()[lag - 1:(T - 2 * lag) + lag - 1, :], ts_W) / (N - 1)
    ts_X_4 = torch.mm(ts_R.clone()[lag:-lag, :], ts_W) / (N - 1)
    ts_X_5 = torch.mm(ts_R.clone()[lag + 1:(T - 2 * lag) + lag + 1, :], ts_W) / (N - 1)

    ts_X = torch.stack([ts_X_1, ts_X_2, ts_X_3, ts_X_4, ts_X_5], dim=0)
    ts_Y = ts_R.clone()[lag:-lag]
    ts_M = ts_M.clone()[lag:-lag]

    mask_train = ts_M.detach().cpu().numpy().reshape(1, -1).T == 0
    mask_test = ts_M.detach().cpu().numpy().reshape(1, -1).T == 1

    dim_X = ts_X.shape[0]
    X_c = ts_X.detach().cpu().numpy().reshape(dim_X, -1).T
    y_c = ts_Y.detach().cpu().numpy().reshape(1, -1).T
    # CCA
    X_c_train = X_c[mask_train[:, 0], :]
    y_c_train = y_c[mask_train[:, 0], :]
    X_c_test = X_c[mask_test[:, 0], :]
    y_c_test = y_c[mask_test[:, 0], :]
    if select == 'cons':
        X_c_train = tensor(X_c_train, dtype=torch.float32, device=device)
        y_c_train = tensor(y_c_train, dtype=torch.float32, device=device)
        X_c_test = tensor(X_c_test, dtype=torch.float32, device=device)
        y_c_test = tensor(y_c_test, dtype=torch.float32, device=device)
        return X_c_train, y_c_train, X_c_test, y_c_test
    elif select == 'vary':
        def generate_spline_basis(T, m=m_spline, knots_sep=knots_sep):
            knots = np.arange(-1, T, knots_sep)
            knots = knots[1:]  # remove the first knots
            spline_basis = dmatrix(f'bs(x, knots={knots.tolist()}, degree={m - 1}, include_intercept=False)',
                                   {'x': np.arange(T)}, return_type='dataframe')
            return spline_basis

        # Construct base functions and lag terms
        sb = generate_spline_basis(ts_M.shape[0])
        dim_sb = sb.shape[1]
        ts_sb = tensor(sb.values, dtype=torch.float32, device=device)
        ts_sb = ts_sb.unsqueeze(2).expand(-1, -1, ts_X.shape[2])
        ts_sb = ts_sb.permute(1, 0, 2)

        # Extend the two tensors
        ts_X2 = ts_X.clone()
        dim_X2 = ts_X2.shape[0]
        a_expanded = ts_X2.unsqueeze(1).expand(-1, dim_sb, -1, -1)
        b_expanded = ts_sb.unsqueeze(0).expand(dim_X2, -1, -1, -1)
        ts_X2 = a_expanded * b_expanded
        ts_X2 = ts_X2.reshape(-1, T - 2 * lag, N)

        # VCA
        X_v = ts_X2.detach().cpu().numpy().reshape(dim_sb * dim_X, -1).T
        y_v = ts_Y.detach().cpu().numpy().reshape(1, -1).T
        # VCA data
        X_v_train = X_v[mask_train[:, 0], :]
        y_v_train = y_v[mask_train[:, 0], :]
        X_v_test = X_v[mask_test[:, 0], :]
        y_v_test = y_v[mask_test[:, 0], :]

        X_v_train = tensor(X_v_train, dtype=torch.float32, device=device)
        y_v_train = tensor(y_v_train, dtype=torch.float32, device=device)
        X_v_test = tensor(X_v_test, dtype=torch.float32, device=device)
        y_v_test = tensor(y_v_test, dtype=torch.float32, device=device)
        return X_v_train, y_v_train, X_v_test, y_v_test
    else:
        raise ValueError('select must be one of cons or vary')

# function: restore the predicted data to the original data
def hat2tensor(m_y, hat, m_indicator):
    out = m_y.clone()
    back = m_y.clone()

    back = back[lag:-lag].reshape(-1, 1)
    mask = m_indicator[lag:-lag].reshape(-1, 1) == 1
    back[mask[:, 0], :] = hat
    back = back.reshape(-1, num_features)
    out[lag:-lag] = back
    out = out.reshape(-1, n_steps, num_features)
    out = torch.nan_to_num(out)
    return out

# class: discriminator
class NetDisc(nn.Module):  # 判别器
    def __init__(self):
        super(NetDisc, self).__init__()
        # conv1
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)  # batch normalization
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # max pooling

        # conv2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # conv3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # FC
        self.fc1 = nn.Linear(128 * 3, 64)
        self.dropout = nn.Dropout(0.5)  # Dropout
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input: (batch_size, 24, 12)
        x = x.permute(0, 2, 1)  # adjust (batch_size, 12, 24)

        # conv1
        x = self.conv1(x)  # (batch_size, 32, 24)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch_size, 32, 12)

        # conv2
        x = self.conv2(x)  # (batch_size, 64, 12)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch_size, 64, 6)

        # conv3
        x = self.conv3(x)  # (batch_size, 128, 6)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)  # (batch_size, 128, 3)

        # FC
        x = x.flatten(start_dim=1)  # flatten (batch_size, 128*3)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu4(x)
        x = self.fc2(x)  # (batch_size, 12)
        x = self.sigmoid(x)

        # adjust (batch_size, 24, 12)
        x = x.unsqueeze(1).repeat(1, 24, 1)
        return x

    def loss(self, outputs, labels):
        return nn.BCELoss()(outputs, labels)

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            return self(x)

# function: discriminator
def discriminator(X, Y):  # X (hat): (batch_size, 24, 12), Y (indicator): (batch_size, 24, 12)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    set_my_seed()
    model = NetDisc()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs_D):
        model.train()
        for batch_X, batch_Y in dataloader:
            outputs = model(batch_X)  # (batch_size, 24, 12)
            loss = model.loss(outputs, batch_Y)
            optimizer.zero_grad() # optimize
            loss.backward()
            optimizer.step()
    return model

# class: regression
class NetReg(nn.Module):
    def __init__(self, dim_input, X_test, m_data, t_indicator, m_indicator, loss_type, model_disc=None):
        super(NetReg, self).__init__()
        self.linear = nn.Linear(dim_input, 1)
        self.X_test = X_test
        self.t_indicator = t_indicator.float()
        self.loss_type = loss_type
        self.m_data = m_data
        self.m_indicator = m_indicator
        self.model_disc = model_disc

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        return self.linear(x)

    def fit(self, X_train, y_train):
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=y_train.shape[0], shuffle=False)
        set_my_seed()
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(epochs_R):
            self.train()
            for batch_X, batch_y in dataloader:
                outputs_train_hat = self(batch_X)
                loss = self.loss(batch_y, outputs_train_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return loss.item()

    def loss(self, outputs_true, outputs_hat, l_lambda=-0.1):
        loss1 = nn.MSELoss()(outputs_true, outputs_hat)
        if self.loss_type == 1:  # without discriminator
            loss2 = 0
        elif self.loss_type == 2:  # with discriminator
            # Start by generating complete data based on the current regression model
            output_test_hat = self(self.X_test)
            t_outputs_test_hat = hat2tensor(self.m_data, output_test_hat, self.m_indicator)
            # The discriminator discriminates the current complete data
            self.model_disc.eval()
            t_indicator_hat = self.model_disc.predict(t_outputs_test_hat)
            loss2 = self.model_disc.loss(t_indicator_hat, self.t_indicator)
        else:
            raise ValueError('loss_type must be 1 or 2')
        if loss2 == 0:
            return loss1
        while True:
            if  abs(loss2 * l_lambda) < abs(2 * loss1 / 10):
                l_lambda -= 0.1
            else:
                break
        return loss1 + loss2 * l_lambda

    def predict(self, X):
        with torch.no_grad():
            self.eval()
            return self(X)

# function: iterations for test
def iterations(e_t_data_ori, e_t_indicator, e_m_data, e_m_indicator, base_algo, miss_type, miss_rate):
    dict_algo = {1: 'MICE', 2: 'KNN', 3: 'LOCF', 4: 'GRUD', 5: 'USGAN', 6: 'SAITS', 7: 'iTransformer'}
    base_algo = dict_algo[base_algo]
    if base_algo == 'MICE':
        e_t_hat0 = t_hat_train_mice.clone()
        e_mse0 = loss_train_mice
    elif base_algo == 'KNN':
        e_t_hat0 = t_hat_train_knn.clone()
        e_mse0 = loss_train_knn
    elif base_algo == 'LOCF':
        e_t_hat0 = t_hat_train_locf.clone()
        e_mse0 = loss_train_locf
    elif base_algo == 'GRUD':
        e_t_hat0 = t_hat_train_grud.clone()
        e_mse0 = loss_train_grud
    elif base_algo == 'USGAN':
        e_t_hat0 = t_hat_train_usgan.clone()
        e_mse0 = loss_train_usgan
    elif base_algo == 'SAITS':
        e_t_hat0 = t_hat_train_saits.clone()
        e_mse0 = loss_train_saits
    elif base_algo == 'iTransformer':
        e_t_hat0 = t_hat_train_itransformer.clone()
        e_mse0 = loss_train_itransformer
    else:
        raise ValueError('base_algo must be one of MICE, KNN, LOCF, GRUD, USGAN, SAITS, iTransformer')

    # CONS(CCA)
    e_t_hat_i = e_t_hat0.clone()
    last_loss = initial_loss
    for it in tqdm(range(epochs_iter), desc=f'Type{miss_type} Rate{miss_rate} {base_algo} CONS'):
        (X_c_train, y_c_train, X_c_test, y_c_test) = get_cons_vary(e_t_hat_i, e_m_indicator, 'cons')
        # Train a regression model
        model_reg_c = NetReg(dim_input=X_c_train.shape[1], X_test=X_c_test, m_data=e_m_data, t_indicator=e_t_indicator,
                             m_indicator=e_m_indicator, loss_type=1)
        model_reg_c.to(device)
        loss_train = model_reg_c.fit(X_c_train, y_c_train)
        # testing
        y_test_hat = model_reg_c.predict(X_c_test) # Update iterations
        e_t_hat_i = hat2tensor(e_m_data, y_test_hat, e_m_indicator) # Update iterations
        mse = calc_mse(torch.nan_to_num(e_t_hat_i), torch.nan_to_num(e_t_data_ori), e_t_indicator)

        if abs(last_loss - loss_train) <= ksi_:
            break
        last_loss = loss_train
    loss_c_ = mse.item()

    # CONS-GAN(CCAAN)
    last_loss = initial_loss
    for it in tqdm(range(epochs_iter//2), desc=f'Type{miss_type} Rate{miss_rate} {base_algo} CONS-GAN'):
        (X_c_train, y_c_train, X_c_test, y_c_test) = get_cons_vary(e_t_hat_i, e_m_indicator, 'cons')
        model_disc = discriminator(tensor(e_t_hat_i, dtype=torch.float32).to(device),
                                   tensor(e_t_indicator, dtype=torch.float32).to(device))

        model_reg_c = NetReg(dim_input=X_c_train.shape[1], X_test=X_c_test, m_data=e_m_data, t_indicator=e_t_indicator,
                             m_indicator=e_m_indicator, loss_type=2, model_disc=model_disc)
        model_reg_c.to(device)
        loss_train = model_reg_c.fit(X_c_train, y_c_train)

        y_test_hat = model_reg_c.predict(X_c_test)
        e_t_hat_i = hat2tensor(e_m_data, y_test_hat, e_m_indicator)
        mse = calc_mse(torch.nan_to_num(e_t_hat_i), torch.nan_to_num(e_t_data_ori), e_t_indicator)

        if abs(last_loss - loss_train) <= ksi_:
            break
        last_loss = loss_train
    loss_c__ = mse.item()

    # VARY(VCA)
    e_t_hat_i = e_t_hat0.clone()
    last_loss = initial_loss
    for it in tqdm(range(epochs_iter), desc=f'Type{miss_type} Rate{miss_rate} {base_algo} VARY'):
        (X_v_train, y_v_train, X_v_test, y_v_test) = get_cons_vary(e_t_hat_i, e_m_indicator, 'vary')

        model_reg_v = NetReg(dim_input=X_v_train.shape[1], X_test=X_v_test, m_data=e_m_data, t_indicator=e_t_indicator,
                             m_indicator=e_m_indicator, loss_type=1)
        model_reg_v.to(device)
        loss_train = model_reg_v.fit(X_v_train, y_v_train)

        y_test_hat = model_reg_v.predict(X_v_test)
        e_t_hat_i = hat2tensor(e_m_data, y_test_hat, e_m_indicator)
        mse = calc_mse(torch.nan_to_num(e_t_hat_i), torch.nan_to_num(e_t_data_ori), e_t_indicator)
        if abs(last_loss - loss_train) <= ksi_:
            break
        last_loss = loss_train
    loss_v_ = mse.item()

    # VARY-GAN(VCAAN)
    last_loss = initial_loss
    for it in tqdm(range(epochs_iter//2), desc=f'Type{miss_type} Rate{miss_rate} {base_algo} VARY-GAN'):
        (X_v_train, y_v_train, X_v_test, y_v_test) = get_cons_vary(e_t_hat_i, e_m_indicator, 'vary')
        model_disc = discriminator(tensor(e_t_hat_i, dtype=torch.float32).to(device),
                                   tensor(e_t_indicator, dtype=torch.float32).to(device))

        model_reg_v = NetReg(dim_input=X_v_train.shape[1], X_test=X_v_test, m_data=e_m_data, t_indicator=e_t_indicator,
                             m_indicator=e_m_indicator, loss_type=2, model_disc=model_disc)
        model_reg_v.to(device)
        loss_train = model_reg_v.fit(X_v_train, y_v_train)
        y_test_hat = model_reg_v.predict(X_v_test) # Update iterations
        e_t_hat_i = hat2tensor(e_m_data, y_test_hat, e_m_indicator) # Update iterations

        mse = calc_mse(torch.nan_to_num(e_t_hat_i), torch.nan_to_num(e_t_data_ori), e_t_indicator)
        if abs(last_loss - loss_train) <= ksi__:
            break
        last_loss = loss_train
    loss_v__ = mse.item()
    print(f'RAW: {e_mse0:.4f}')
    print(f'CCA: {loss_c_:.4f}, CCAAN: {loss_c__:.4f}\nVCA: {loss_v_:.4f}, VCAAN: {loss_v__:.4f}\n')
    return e_mse0, loss_c_, loss_v_, loss_c__, loss_v__


# Pre-defined parameters
lag = 1  # the lag of the time series
knots_sep = 24  # the separation of the knots, using 24 hours as a unit
start_day = 0  # start day 0
use_days = 30  # use 30 days
epochs_iter = 10  # the number of iterations
epochs_D = 50  # the number of epochs for the discriminator
epochs_R = 50  # the number of epochs for the regression model
epochs_BASE = 100  # the number of epochs for the baseline models
ksi_ = 0.0001  # the stopping condition
ksi__ = 0.0001  # the stopping condition
l_lambda = -0.1  # Loss weight
batch_size = 32  # batch size
m_spline = 5  # the order of the spline basis
initial_loss = 100 # initial loss (large enough)

list_mse = []
for miss_type in [1,2,3,4,5]:  # missing type
    list_mse_type = []
    for miss_rate in [1,2,3]:  # missing rate
        list_mse_rate = []
        data, data_ori, num_features, n_steps, num_samples = run_load_data() # load data
        data = run_make_miss(data, miss_type, miss_rate) # make missing
        print(f'Type{miss_type} Rate{miss_rate}\t True missing rate of data: {torch.isnan(data).sum().item() / data.numel():.4f}')

        # Split data and indicator
        (I_train, I_val, I_test,
         t_data_train, t_data_val, t_data_test,
         t_data_ori_train, t_data_ori_val, t_data_ori_test,
         t_indicator_train, t_indicator_val, t_indicator_test,
         m_data_train, m_data_val, m_data_test,
         m_data_ori_train, m_data_ori_val, m_data_ori_test,
         m_indicator_train, m_indicator_val, m_indicator_test) = run_split_data_and_indicator(data, data_ori,
                                                                                              num_features,
                                                                                              num_samples)
        # Baselines
        (t_hat_train_mice, loss_train_mice,
         t_hat_train_knn, loss_train_knn,
         t_hat_train_locf, loss_train_locf,
         t_hat_train_grud, loss_train_grud,
         t_hat_train_usgan, loss_train_usgan,
         t_hat_train_saits, loss_train_saits,
         t_hat_train_itransformer, loss_train_itransformer) = run_baselines(t_data_train, t_data_val, t_data_ori_train,
                                                                            t_data_ori_val, t_indicator_train,
                                                                            m_data_train)
        print(f'mice: {loss_train_mice:.4f},\n'
              f'knn: {loss_train_knn:.4f},\n'
              f'locf: {loss_train_locf:.4f},\n'
              f'grud: {loss_train_grud:.4f},\n'
              f'usgan: {loss_train_usgan:.4f},\n'
              f'saits: {loss_train_saits:.4f},\n'
              f'itransformer: {loss_train_itransformer:.4f}')
        for base_algo in [1,2,3,4,5,6,7]:  # Baseline models
            # Initialize the data
            # e_t_data = t_data_train.clone()  # It doesn't seem to work
            e_t_data_ori = t_data_ori_train.clone()
            e_t_indicator = t_indicator_train.clone()
            e_m_data = m_data_train.clone()
            # e_m_data_ori = m_data_ori_train.clone()  # It doesn't seem to work
            e_m_indicator = m_indicator_train.clone()
            list_mse_algo = iterations(e_t_data_ori, e_t_indicator, e_m_data, e_m_indicator, base_algo, miss_type,
                                       miss_rate)
            list_mse_rate.append(list_mse_algo)
        list_mse_type.append(list_mse_rate)
    list_mse.append(list_mse_type)

df = pd.DataFrame(np.array(list_mse).reshape(-1, 5), columns=['Baseline', 'CCA', 'VCA', 'CCAAN', 'VCAAN'])
df.to_excel(f'./res/RES_({start_day}_use{use_days}).xlsx', index=False)
end_time = time.time()
print('Time:', time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))
