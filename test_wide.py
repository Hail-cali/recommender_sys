import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from model.wide_deep import *

# from torchfm.dataset.movielens import MovieLens1MDataset
from torch.utils.data import DataLoader
import tqdm
from collections import OrderedDict


class BookDataLoader(torch.utils.data.Dataset):

    def __init__(self, dataset_path, sep='::', engine='python', header=None):
        df = pd.read_csv(dataset_path, engine=engine).sort_values(by='User-ID')
        df = df[df['Book-Rating'] > 0]
        df, self.user_key, self.item_key = self.indexing_df(df)

        data = df.to_numpy()[:, :3]

        self.items = data[:, :2].astype(np.int)
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1  # index start from 0, shape need to be sum 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def indexing_df(self, df):
        user_key = OrderedDict()
        book_key = OrderedDict()

        if 'ISBN' in df.columns:
            idx = df['ISBN'].unique()
            book_key = OrderedDict(zip(idx, range(len(idx))))
            df.loc[:,'ISBN'] = df['ISBN'].apply(lambda x: book_key[x])

        if 'User-ID' in df.columns:
            idx = df['User-ID'].unique()
            user_key = OrderedDict(zip(idx, range(len(idx))))
            df.loc[:,'User-ID'] = df['User-ID'].apply(lambda x: user_key[x])

        return df, user_key, book_key

    def __preprocess_target(self, target):
        self.mu = np.mean(target)
        # target[target <= 3] = 0
        # target[target > 3] = 1
        return target - self.mu


class BookAddDataLoader(torch.utils.data.Dataset):

    def __init__(self, dataset_path, sep='::', engine='python', header=None):

        col = ['User-ID', 'ISBN', 'Age', 'age_bin', 'eigen_bin',
       'eigen_rated', 'deg_rated', 'eigen_imp', 'deg_imp', 'Book-Rating']
        choose = ['User-ID', 'ISBN','age_bin','eigen_bin', 'Book-Rating']
        df = pd.read_csv(dataset_path, engine=engine).sort_values(by='User-ID').loc[:, choose]

        df = df[df['Book-Rating'] > 0]
        df, self.user_key, self.item_key = self.indexing_df(df)

        data = df.to_numpy()[:, : len(choose)]

        self.items = data[:, :len(choose)-1].astype(np.int)
        self.targets = self.__preprocess_target(data[:, -1]).astype(np.float32)

        self.field_dims = np.max(self.items, axis=0) + 1  # index start from 0, shape need to be sum 1
        self.field_dims_new = self.shape_field(df, choose[:-1])
        print(f'old {self.field_dims} dtype:: {type(self.field_dims)}')
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def indexing_df(self, df):
        user_key = OrderedDict()
        book_key = OrderedDict()

        if 'ISBN' in df.columns:
            idx = df['ISBN'].unique()
            book_key = OrderedDict(zip(idx, range(len(idx))))
            df.loc[:,'ISBN'] = df['ISBN'].apply(lambda x: book_key[x])

        if 'User-ID' in df.columns:
            idx = df['User-ID'].unique()
            user_key = OrderedDict(zip(idx, range(len(idx))))
            df.loc[:,'User-ID'] = df['User-ID'].apply(lambda x: user_key[x])

        return df, user_key, book_key

    def shape_field(self, df, choose):
        field = []
        for col in choose:
            if col in ['User-ID', 'ISBN','age_bin']:
                field.append(len(df[col].unique()))
            else:
                field.append(1)

        return np.array(field, dtype=np.int)


    def __preprocess_target(self, target):
        self.mu = np.mean(target)
        # target[target <= 3] = 0
        # target[target > 3] = 1
        return target - self.mu

class MovieDataLodaer(torch.utils.data.Dataset):

    def __init__(self, dataset_path, sep='::', engine='python', header=None):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        self.mu = np.mean(target)
        # target[target <= 3] = 0
        # target[target > 3] = 1
        return target - self.mu



class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_rmse = 3
        self.save_path = save_path

    def is_continuable(self, model, rmse):
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def get_dataset(dataset_name, path):

    if dataset_name == 'movielens1M':

        return MovieDataLodaer(path)
    elif dataset_name == 'book':

        return BookDataLoader(path)

    elif dataset_name == 'bookadd':
        return BookAddDataLoader(path)


def get_model(model_name, field_dims):
    if model_name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=32, mlp_dims=(16, 16), dropout=0.1)

    elif model_name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))

    elif model_name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=64, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        # loss = criterion(y, target.float())

        model.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, mu, device):
    model.eval()
    targets, predicts = list(), list()
    rmse = RMSELoss()

    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target)
            predicts.extend(y)

        print(torch.tensor(predicts).float())

    return rmse(torch.tensor(predicts).float(), torch.tensor(targets).float())


def predict(model, data_loader, mu , device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields) + mu
            targets.extend(target + mu)
            predicts.extend(y)

    return torch.tensor(predicts).float()



def main(dataset_name, dataset_path, model_name, epoch, learning_rate,
         batch_size, weight_decay, gpu_num,save_dir):


    path = 'ml-1m/ratings.dat'
    d_path = 'book_review/BX-Book-Ratings.csv'

    try:
        device = torch.device(f'gpu:{gpu_num}' if torch.cuda.is_available() else f'cpu')

    except RuntimeError:
        device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else f'cpu')

    # dataset = BookDataLodaer(d_path)
    # dataset = MovieDataLodaer(path)
    dataset = get_dataset(dataset_name, dataset_path)

    field_dims = dataset.field_dims

    print(f'dataset :: {dataset_name} size :: {dataset.field_dims} path :: {dataset_path}')
    print(f'device setting:: {device}')

    train_length = int(len(dataset) * 0.7)
    print(f'train length {train_length}')
    test_length = int(len(dataset) * 0.3)
    valid_length = len(dataset) - train_length - test_length

    train_dataset, test_dataset, valid_dataset,  = torch.utils.data.random_split(
        dataset, (train_length, test_length, valid_length, ))


    mu = torch.tensor(dataset.mu).to(device)
    try:
        print(f'mu : {mu}')

    except:
        print()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)


    # model = NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    # model = WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)

    model = get_model(model_name, field_dims)
    print(f'model name: {model_name}')
    model.to(device)

    # criterion = torch.nn.BCELoss()
    criterion = RMSELoss()
    # criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        rmse = test(model, test_data_loader, mu, device)
        print('epoch:', epoch_i, 'test: rmse:', rmse)
        if not early_stopper.is_continuable(model, rmse):
            print(f'test: best rmse: {early_stopper.best_rmse}')
            break

    # rmse = test(model, test_data_loader, mu, device)
    # print(f'test rmse: {rmse}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default= 'bookadd',
                        help=' movielens1M | book | bookadd ')

    parser.add_argument('--dataset_path', default='book_review/total.csv',
                        help=' ml-1m/ratings.dat  |  book_review/BX-Book-Ratings.csv | book_review/total.csv ')

    parser.add_argument('--model_name', default='xdfm')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1028)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device_num', type=str, default=1)
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()

    PATH = '/Users/george/testData/book'
    RAT = 'BX-Book-Ratings.csv'
    USR = 'BX-Users.csv'
    BOK = 'BX-Books.csv'


    main(
        args.dataset_name, args.dataset_path, args.model_name, args.epoch,
        args.learning_rate, args.batch_size, args.weight_decay, args.device_num, args.save_dir)