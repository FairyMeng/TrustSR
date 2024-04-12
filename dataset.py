import pandas as pd
import numpy as np
import torch


class Dataset(object):
    def __init__(self, path, sep=',', session_key='SessionID', item_key='ItemID', time_key='Time', n_sample=-1, itemmap=None, itemstamp=None, time_sort=False):
        # Read csv
        self.df = pd.read_csv(path, sep=sep, dtype={session_key: int, item_key: int, time_key: float})
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        if n_sample > 0:
            self.df = self.df[:n_sample]

        # Add colummn item index to data
        self.add_item_indices(itemmap=itemmap)
        self.df.sort_values([session_key, time_key], inplace=True)
        self.click_offsets = self.get_click_offset()
        self.session_idx_arr = self.order_session_idx()

    def add_item_indices(self, itemmap=None):
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # type is numpy.ndarray
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            # Build itemmap is a DataFrame that have 2 columns (self.item_key, 'item_idx)
            itemmap = pd.DataFrame({self.item_key: item_ids,
                                   'item_idx': item2idx[item_ids].values})
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')

    def get_click_offset(self):
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets

    def order_session_idx(self):
        if self.time_sort:
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())
        return session_idx_arr

    @property
    def items(self):
        return self.itemmap[self.item_key].unique()


class DataLoader():
    def __init__(self, dataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # initializations
        df = self.dataset.df
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = click_offsets[session_idx_arr[iters]]
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            minlen = (end - start).min()
            # Item indices(for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]

            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                input = torch.LongTensor(idx_input)
                target = torch.LongTensor(idx_target)
                yield input, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]

