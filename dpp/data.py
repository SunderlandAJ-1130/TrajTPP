from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.utils.data as data_utils
from pathlib import Path
import pickle


# dataset_dir = Path(__file__).parents[2] / 'data'

class Batch():
    def __init__(self, in_time, out_time, length, index=None, in_mark=None, out_mark=None,
                 in_driver=None, out_driver=None, in_timeofday=None, out_timeofday=None,
                 in_minute=None, out_minute=None, in_dayofweek=None, out_dayofweek=None):
        self.in_time = in_time
        self.out_time = out_time
        self.length = length
        self.index = index
        self.in_mark = in_mark.long()
        self.out_mark = out_mark.long()
        self.in_driver = in_driver.long()
        self.out_driver = out_driver.long()
        self.in_timeofday = in_timeofday.long()
        self.out_timeofday = out_timeofday.long()
        self.in_minute = in_minute.long()
        self.out_minute = out_minute.long()
        self.in_dayofweek = in_dayofweek.long()
        self.out_dayofweek = out_dayofweek.long()
        

class SimpleBatch():
    def __init__(self, delta_times, marks, hours=None, drivers=None, minutes=None, dayofweek=None, device='cuda:0'):
        self.device = device
        
        self.in_time = [torch.tensor(t, device=self.device) for t in delta_times]
        self.out_time = [torch.tensor(t, device=self.device) for t in delta_times]
        # self.in_time = [t.log_() for t in self.in_time]
        
        self.in_time = torch.stack(self.in_time).reshape(len(delta_times), -1).to(self.device)
        
        self.in_mark = marks
        self.out_mark = marks
            
        self.in_driver = drivers
        self.out_driver = drivers

        self.in_timeofday = hours
        self.out_timeofday = hours
        
        self.in_minute = minutes
        self.out_minute = minutes
        
        self.in_dayofweek = dayofweek
        self.out_dayofweek = dayofweek
        
        length = [len(t) for t in self.in_time]
        self.length = torch.Tensor(length, device=self.device)
        

def load_dataset(name, normalize_min_max=False, log_mode=True, device='cuda:0'):
    """Load dataset."""
    if not name.endswith('.pkl'):
        name += '.pkl'
    
    file = open(name, 'rb')
    loader = pickle.load(file)
    print(loader.keys())
    # loader = np.load(name, allow_pickle=True)

    # loader = dict(np.load(name, allow_pickle=True))
    arrival_times = loader['arrival_times']
    marks = loader.get('marks')
    drivers = loader.get('drivers')
    timeofdays = loader.get('timeofdays')
    num_drivers = loader.get('num_drivers')
    minutes = loader.get('minutes')
    dayofweeks = loader.get('dayofweeks')
    
    num_classes = len({x for s in marks for x in s}) if marks is not None else 1
    delta_times = [np.concatenate([[1.0], np.ediff1d(time)]) for time in arrival_times]
    
    return SequenceDataset(
        delta_times, marks=marks, num_classes=num_classes, log_mode=log_mode, dayofweeks=dayofweeks,
        device=device, drivers=drivers, timeofdays=timeofdays, minutes=minutes), num_drivers


def collate(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    in_time = [item[0] for item in batch]
    out_time = [item[1] for item in batch]
    in_marks = [item[2] for item in batch]
    out_marks = [item[3] for item in batch]
    in_drivers = [item[4] for item in batch]
    out_drivers = [item[5] for item in batch]
    in_timeofdays = [item[6] for item in batch]
    out_timeofdays = [item[7] for item in batch]
    in_minutes = [item[8] for item in batch]
    out_minutes = [item[9] for item in batch]
    in_dayofweeks = [item[10] for item in batch]
    out_dayofweeks = [item[11] for item in batch]

    index = torch.tensor([item[12] for item in batch], device=in_time[0].device)
    length = torch.Tensor([len(item) for item in in_time], device=in_time[0].device)
    
    in_time = torch.nn.utils.rnn.pad_sequence(in_time, batch_first=True)
    out_time = torch.nn.utils.rnn.pad_sequence(out_time, batch_first=True)
    in_mark = torch.nn.utils.rnn.pad_sequence(in_marks, batch_first=True)
    out_mark = torch.nn.utils.rnn.pad_sequence(out_marks, batch_first=True)
    in_driver = torch.nn.utils.rnn.pad_sequence(in_drivers, batch_first=True)
    out_driver = torch.nn.utils.rnn.pad_sequence(out_drivers, batch_first=True)
    in_timeofday = torch.nn.utils.rnn.pad_sequence(in_timeofdays, batch_first=True)
    out_timeofday = torch.nn.utils.rnn.pad_sequence(out_timeofdays, batch_first=True)
    in_minute = torch.nn.utils.rnn.pad_sequence(in_minutes, batch_first=True)
    out_minute = torch.nn.utils.rnn.pad_sequence(out_minutes, batch_first=True)
    in_dayofweek = torch.nn.utils.rnn.pad_sequence(in_dayofweeks, batch_first=True)
    out_dayofweek = torch.nn.utils.rnn.pad_sequence(out_dayofweeks, batch_first=True)

    in_time[:, 0] = 0 # set first to zeros

    return Batch(in_time, out_time, length, in_mark=in_mark, out_mark=out_mark, index=index, in_minute=in_minute, out_minute=out_minute,
                 in_driver=in_driver, out_driver=out_driver, in_timeofday=in_timeofday, out_timeofday=out_timeofday, 
                 in_dayofweek=in_dayofweek, out_dayofweek=out_dayofweek)


class SequenceDataset(data_utils.Dataset):
    """Dataset class containing variable length sequences.

    Args:
        delta_times: Inter-arrival times between events. List of variable length sequences.

    """
    def __init__(self, delta_times=None, marks=None, in_times=None, out_times=None, device='cuda:0',
                 in_marks=None, out_marks=None, index=None, log_mode=True, num_classes=1, dayofweeks=None,
                 drivers=None, timeofdays=None, minutes=None):
        # sourcery skip: low-code-quality
        self.num_classes = num_classes
        self.device = device
        
        if delta_times is not None:
            self.in_times = [torch.Tensor(t[:-1], device=self.device) for t in delta_times]
            self.out_times = [torch.Tensor(t[1:], device=self.device) for t in delta_times]
        else:
            if (not all(torch.is_tensor(t) for t in in_times) or
                not all(torch.is_tensor(t) for t in out_times)):
                raise ValueError("in and out times and marks must all be torch.Tensors")
            self.in_times = in_times
            self.out_times = out_times

        if marks is not None:
            self.in_marks = [torch.Tensor(m[:-1], device=self.device) for m in marks]
            self.out_marks = [torch.Tensor(m[1:], device=self.device) for m in marks]
        else:
            self.in_marks = None
            self.out_marks = None
            
        if drivers is not None:
            self.in_drivers = [torch.tensor(m[:-1], device=self.device) for m in drivers]
            self.out_drivers = [torch.tensor(m[1:], device=self.device) for m in drivers]
        else:
            self.in_drivers = None
            self.out_drivers = None
        
        if timeofdays is not None:
            self.in_timeofdays = [torch.tensor(m[:-1], device=self.device) for m in timeofdays]
            self.out_timeofdays = [torch.tensor(m[1:], device=self.device) for m in timeofdays]
        else:
            self.in_timeofdays = None
            self.out_timeofdays = None
            
        if dayofweeks is not None:
            self.in_dayofweeks = [torch.tensor(m[:-1], device=self.device) for m in dayofweeks]
            self.out_dayofweeks = [torch.tensor(m[1:], device=self.device) for m in dayofweeks]
        else:
            self.in_dayofweeks = None
            self.out_dayofweeks = None
            
        if minutes is not None:
            self.in_minutes = [torch.tensor(m[:-1], device=self.device) for m in minutes]
            self.out_minutes = [torch.tensor(m[1:], device=self.device) for m in minutes]
        else:
            self.in_minutes = None
            self.out_minutes = None
            
        if index is None:
            index = torch.arange(len(self.in_times), device=self.device)
        if not torch.is_tensor(index):
            index = torch.tensor(index, device=self.device)

        if self.in_marks is None:
            self.in_marks = [torch.zeros_like(x, device=self.device) for x in self.in_times]
            self.out_marks = [torch.zeros_like(x, device=self.device) for x in self.out_times]

        self.index = index
        self.validate_times()
        # Apply log transformation to inputs
        if log_mode:
            self.in_times = [t.log_() for t in self.in_times]

    @property
    def num_series(self):
        return len(self.in_times)

    def validate_times(self):
        if len(self.in_times) != len(self.out_times):
            raise ValueError("in_times and out_times have different lengths.")

        if len(self.index) != len(self.in_times):
            raise ValueError("Length of index should match in_times/out_times")

        for s1, s2, s3, s4 in zip(self.in_times, self.out_times, self.in_marks, self.out_marks):
            if len(s1) != len(s2) or len(s3) != len(s4):
                raise ValueError("Some in/out series have different lengths.")
            # if s3.max() >= self.num_classes or s4.max() >= self.num_classes:
            #     raise ValueError("Marks should not be larger than number of classes.")

    def break_down_long_sequences(self, max_seq_len):
        """Break down long sequences into shorter sub-sequences."""
        self.validate_times()
        new_in_times = []
        new_out_times = []
        new_in_marks = []
        new_out_marks = []
        new_index = []
        new_in_drivers = []
        new_out_drivers = []
        new_in_timeofdays = []
        new_out_timeofdays = []
        new_in_minutes = []
        new_out_minutes = []
        new_in_dayofweeks = []
        new_out_dayofweeks = []
        
        for idx in range(self.num_series):
            current_in = self.in_times[idx]
            current_out = self.out_times[idx]
            current_in_mark = self.in_marks[idx]
            current_out_mark = self.out_marks[idx]
            current_in_driver = self.in_drivers[idx]
            current_out_driver = self.out_drivers[idx]
            current_in_timeofday = self.in_timeofdays[idx]
            current_out_timeofday = self.out_timeofdays[idx]
            current_in_minute = self.in_minutes[idx]
            current_out_minute = self.out_minutes[idx]
            current_in_dayofweek = self.in_dayofweeks[idx]
            current_out_dayofweek = self.out_dayofweeks[idx]
            
            num_batches = int(np.ceil(len(current_in) / max_seq_len))
            for b in range(num_batches):
                new_in = current_in[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_times.append(new_in)
                new_out = current_out[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_times.append(new_out)

                new_in_mark = current_in_mark[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_marks.append(new_in_mark)
                new_out_mark = current_out_mark[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_marks.append(new_out_mark)
                
                new_in_driver = current_in_driver[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_drivers.append(new_in_driver)
                new_out_driver = current_out_driver[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_drivers.append(new_out_driver)
                
                new_in_timeofday = current_in_timeofday[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_timeofdays.append(new_in_timeofday)
                new_out_timeofday = current_out_timeofday[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_timeofdays.append(new_out_timeofday)
                
                new_in_minute = current_in_minute[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_minutes.append(new_in_minute)
                new_out_minute = current_out_minute[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_minutes.append(new_out_minute)
                
                new_in_dayofweek = current_in_dayofweek[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_dayofweeks.append(new_in_dayofweek)
                new_out_dayofweek = current_out_dayofweek[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_dayofweeks.append(new_out_dayofweek)

                new_index.append(self.index[idx])
        self.in_times = new_in_times
        self.out_times = new_out_times
        self.in_marks = new_in_marks
        self.out_marks = new_out_marks
        self.in_drivers = new_in_drivers
        self.out_drivers = new_out_drivers
        self.in_timeofdays = new_in_timeofdays
        self.out_timeofdays = new_out_timeofdays
        self.in_minutes = new_in_minutes
        self.out_minutes = new_out_minutes
        self.in_dayofweeks = new_in_dayofweeks
        self.out_dayofweeks = new_out_dayofweeks
        self.index = torch.tensor(new_index, device=self.device)
        self.validate_times()
        return self

    def train_val_test_split_whole(self, train_size=0.6, val_size=0.2, test_size=0.2, seed=123):
        """Split dataset into train, val and test parts."""
        np.random.seed(seed)
        all_idx = np.arange(self.num_series)
        train_idx, val_test_idx = train_test_split(all_idx, train_size=train_size, test_size=(val_size + test_size))
        if val_size == 0:
            val_idx = []
            test_idx = val_test_idx
        else:
            val_idx, test_idx = train_test_split(val_test_idx, train_size=(val_size / (val_size + test_size)),
                                                 test_size=(test_size / (val_size + test_size)))

        def get_dataset(ind):
            in_time, out_time = [], []
            in_mark, out_mark = [], []
            index = []
            for i in ind:
                in_time.append(self.in_times[i])
                out_time.append(self.out_times[i])
                in_mark.append(self.in_marks[i])
                out_mark.append(self.out_marks[i])
                index.append(self.index[i])
            return SequenceDataset(in_times=in_time, out_times=out_time, in_marks=in_mark, out_marks=out_mark, device=self.in_times[i].device, 
                                   index=index, log_mode=False, num_classes=self.num_classes)

        data_train = get_dataset(train_idx)
        data_val = get_dataset(val_idx)
        data_test = get_dataset(test_idx)

        return data_train, data_val, data_test

    def train_val_test_split_each(self, train_size=0.6, val_size=0.2, test_size=0.2, seed=123):
        """Split each sequence in the dataset into train, val and test parts."""
        np.random.seed(seed)
        in_train, in_val, in_test = [], [], []
        out_train, out_val, out_test = [], [], []
        in_mark_train, in_mark_val, in_mark_test = [], [], []
        out_mark_train, out_mark_val, out_mark_test = [], [], []
        index_train, index_val, index_test = [], [], []
        in_driver_train, in_driver_val, in_driver_test = [], [], []
        out_driver_train, out_driver_val, out_driver_test = [], [], []
        in_timeofday_train, in_timeofday_val, in_timeofday_test = [], [], []
        out_timeofday_train, out_timeofday_val, out_timeofday_test = [], [], []
        in_minute_train, in_minute_val, in_minute_test = [], [], []
        out_minute_train, out_minute_val, out_minute_test = [], [], []
        in_dayofweek_train, in_dayofweek_val, in_dayofweek_test = [], [], []
        out_dayofweek_train, out_dayofweek_val, out_dayofweek_test = [], [], []
        
        for idx in range(self.num_series):
            n_elements = len(self.in_times[idx])
            n_train = int(train_size * n_elements)
            n_val = int(val_size * n_elements)

            if n_train == 0 or n_val == 0 or (n_elements - n_train - n_val) == 0:
                continue

            in_train.append(self.in_times[idx][:n_train])
            in_val.append(self.in_times[idx][n_train : (n_train + n_val)])
            in_test.append(self.in_times[idx][(n_train + n_val):])

            in_mark_train.append(self.in_marks[idx][:n_train])
            in_mark_val.append(self.in_marks[idx][n_train : (n_train + n_val)])
            in_mark_test.append(self.in_marks[idx][(n_train + n_val):])

            out_train.append(self.out_times[idx][:n_train])
            out_val.append(self.out_times[idx][n_train : (n_train + n_val)])
            out_test.append(self.out_times[idx][(n_train + n_val):])

            out_mark_train.append(self.out_marks[idx][:n_train])
            out_mark_val.append(self.out_marks[idx][n_train : (n_train + n_val)])
            out_mark_test.append(self.out_marks[idx][(n_train + n_val):])

            index_train.append(self.index[idx])
            index_val.append(self.index[idx])
            index_test.append(self.index[idx])
            
            in_driver_train.append(self.in_drivers[idx][:n_train])
            in_driver_val.append(self.in_drivers[idx][n_train : (n_train + n_val)])
            in_driver_test.append(self.in_drivers[idx][(n_train + n_val):])
            
            out_driver_train.append(self.out_drivers[idx][:n_train])
            out_driver_val.append(self.out_drivers[idx][n_train : (n_train + n_val)])
            out_driver_test.append(self.out_drivers[idx][(n_train + n_val):])
            
            in_timeofday_train.append(self.in_timeofdays[idx][:n_train])
            in_timeofday_val.append(self.in_timeofdays[idx][n_train : (n_train + n_val)])
            in_timeofday_test.append(self.in_timeofdays[idx][(n_train + n_val):])
            
            out_timeofday_train.append(self.out_timeofdays[idx][:n_train])
            out_timeofday_val.append(self.out_timeofdays[idx][n_train : (n_train + n_val)])
            out_timeofday_test.append(self.out_timeofdays[idx][(n_train + n_val):])
            
            in_minute_train.append(self.in_minutes[idx][:n_train])
            in_minute_val.append(self.in_minutes[idx][n_train : (n_train + n_val)])
            in_minute_test.append(self.in_minutes[idx][(n_train + n_val):])
            
            out_minute_train.append(self.out_minutes[idx][:n_train])
            out_minute_val.append(self.out_minutes[idx][n_train : (n_train + n_val)])
            out_minute_test.append(self.out_minutes[idx][(n_train + n_val):])
            
            in_dayofweek_train.append(self.in_dayofweeks[idx][:n_train])
            in_dayofweek_val.append(self.in_dayofweeks[idx][n_train : (n_train + n_val)])
            in_dayofweek_test.append(self.in_dayofweeks[idx][(n_train + n_val):])
            
            out_dayofweek_train.append(self.out_dayofweeks[idx][:n_train])
            out_dayofweek_val.append(self.out_dayofweeks[idx][n_train : (n_train + n_val)])
            out_dayofweek_test.append(self.out_dayofweeks[idx][(n_train + n_val):])
            
        data_train = SequenceDataset(in_times=in_train, out_times=out_train, in_marks=in_mark_train, out_marks=out_mark_train,
                                     index=index_train, log_mode=False, num_classes=self.num_classes, device=self.device, 
                                     drivers=in_driver_train, timeofdays=in_timeofday_train, minutes=in_minute_train, dayofweeks=in_dayofweek_train)
        data_val = SequenceDataset(in_times=in_val, out_times=out_val, in_marks=in_mark_val, out_marks=out_mark_val,
                                   index=index_val, log_mode=False, num_classes=self.num_classes, device=self.device, 
                                   drivers=in_driver_val, timeofdays=in_timeofday_val, minutes=in_minute_val, dayofweeks=in_dayofweek_val)
        data_test = SequenceDataset(in_times=in_test, out_times=out_test, in_marks=in_mark_test, out_marks=out_mark_test,
                                    index=index_test, log_mode=False, num_classes=self.num_classes, device=self.device, 
                                    drivers=in_driver_test, timeofdays=in_timeofday_test, minutes=in_minute_test, dayofweeks=in_dayofweek_test)
        return data_train, data_val, data_test

    def normalize(self, mean_in=None, std_in=None, std_out=None):
        """Apply mean-std normalization to in_times."""
        if mean_in is None or std_in is None:
            mean_in, std_in = self.get_mean_std_in()
        self.in_times = [(t - mean_in) / std_in for t in self.in_times]
        if std_out is not None:
            # _, std_out = self.get_mean_std_out()
            self.out_times = [t / std_out for t in self.out_times]
        return self

    def get_mean_std_in(self):
        """Get mean and std of in_times."""
        flat_in_times = torch.cat(self.in_times)
        return flat_in_times.mean(), flat_in_times.std()

    def get_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times)
        return flat_out_times.mean(), flat_out_times.std()

    def get_log_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times).log()
        return flat_out_times.mean(), flat_out_times.std()

    def flatten(self):
        """Merge in_times and out_times to a single sequence."""
        flat_in_times = torch.cat(self.in_times)
        flat_out_times = torch.cat(self.out_times)
        return SequenceDataset(in_times=[flat_in_times], out_times=[flat_out_times], log_mode=False, num_classes=self.num_classes)

    def __add__(self, other):
        new_in_times = self.in_times + other.in_times
        new_out_times = self.out_times + other.out_times
        new_index = torch.cat([self.index, other.index + len(self.index)])
        return SequenceDataset(in_times=new_in_times, out_times=new_out_times, index=new_index,
                               num_classes=self.num_classes, log_mode=False)

    def __getitem__(self, key):
        return self.in_times[key], self.out_times[key], self.in_marks[key], self.out_marks[key], \
            self.in_drivers[key], self.out_drivers[key], self.in_timeofdays[key], self.out_timeofdays[key], \
            self.in_minutes[key], self.out_minutes[key], self.in_dayofweeks[key], self.out_dayofweeks[key], self.index[key]

    def __len__(self):
        return self.num_series

    def __repr__(self):
        return f"SequenceDataset({self.num_series})"
