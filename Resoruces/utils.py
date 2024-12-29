import numpy as np 
import pandas as pd 
import torch


class CryptoDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self,csv_file, seq_len, is_train=False):
        super(CryptoDataset, self).__init__()
        self.data_frame = pd.read_csv(csv_file)
        self.is_train = is_train
        self.seq_len = seq_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        time = torch.from_numpy(np.expand_dims(self.data_frame.iloc[idx:(idx+self.seq_len)]['close'].to_numpy(dtype=np.float32),axis=0))
        values = torch.from_numpy(np.array([self.data_frame.iloc[idx:(idx + self.seq_len)]['open'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['close'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['high'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['low'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['volume'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['quote_asset_volume'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['number_of_trades'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['taker_buy_base_volume'].tolist(),
                    self.data_frame.iloc[idx:(idx + self.seq_len)]['taker_buy_quote_volume'].tolist()],dtype=np.float32))
        if self.is_train:
            
            target = torch.tensor([self.data_frame.iloc[idx + self.seq_len]['target']],dtype=torch.float32)
            
            return {'time': time, 'values': values,'target': target}
        else:
            return {'time': time, 'values': values}

    def __len__(self):
        return len(self.data_frame)


class DataUtils:

    # def __init__(self, file_name):
    #     self.file_name = file_name

    def load_csv_to_dataframe(file_name):
        return pd.read_csv(file_name)
    
# class CryptoDirectionDataset(Dataset):

#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Arguments:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.dataset_frame = pd.read_csv(csv_file)
#         self.dataset_frame = self.dataset_frame.drop('row_id')
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         timestamp = io.imread(img_name)
#         #row_id,timestamp,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume
        
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample