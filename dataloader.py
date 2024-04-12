from PIL import Image
import os
from tqdm import tqdm
import torch
from transformers import AutoProcessor, CLIPModel
import h5py
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import random


dataset_path = './datasets/Amazon_Toys_and_Games'

inter_data = 'my_data_meta.csv'
tensor_data = 'tensor_data.hdf5'
image_tensor_data = 'clip_image_tensor_data.hdf5'
text_tensor_data = 'text_tensor_data.hdf5'
inter_path = os.path.join(dataset_path, f'{inter_data}')
tensor_path = os.path.join(dataset_path, f'{tensor_data}')
image_tensor_path = os.path.join(dataset_path, f'{image_tensor_data}')
text_tensor_path = os.path.join(dataset_path, f'{text_tensor_data}')


def Text_to_Image(model, processor, item_name, text):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    image = Image.open('./datasets/Amazon_Toys_and_Games/image/{}.jpg'.format(item_name))
    inputs = processor(
        text=text, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    text_tensor = outputs.text_embeds
    probs = logits_per_image.softmax(dim=1)

    max_score_idx = torch.argmax(probs.T)
    max_score_tensor = text_tensor[max_score_idx]
    max_score_tensor = max_score_tensor.to(device)
    text_embedding = max_score_tensor

    return text_embedding


def dataprocess():

    df_inter = pd.read_csv(inter_path, delimiter=',')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model = model.to(device)

    with h5py.File(text_tensor_path, 'a') as file:
        pbar = tqdm(total=len(df_inter))
        for index, row in df_inter.iterrows():
            key = str(row[4])
            if key in file.keys():
                continue
            else:
                item_review_embedding=Text_to_Image(model, processor, row[2], row[7])
                item_list = item_review_embedding.cpu().detach().numpy()
                file.create_dataset(key, data=item_list)
            if index%10 == 0:    
                pbar.update(10)   
        file.close()
        pbar.close()


def load():
    dataset_path = './datasets/Amazon_Toys_and_Games'
    inter_data = 'user_review.csv'
    save_data = 'TrustSR.csv'
    inter_path = os.path.join(dataset_path, f'{inter_data}')
    save_path = os.path.join(dataset_path, f'{save_data}')
    # get interaction_sequence and correspond review sequence
    inter_usecols = ['user_id', 'item_id', 'timestamp']
    inter_dtype = {'user_id': str, 'item_id': str, 'timestamp': int}
    with open(inter_path, 'r') as f:  # loda interaction data
        df_inter = pd.read_csv(inter_path, delimiter=',', usecols=inter_usecols, dtype=inter_dtype)
        df_inter['new_user_id'], user_map = pd.factorize(df_inter['user_id'])
        df_inter['new_item_id'], item_map = pd.factorize(df_inter['item_id'])

    df_inter['new_user_id'] = df_inter['new_user_id'].astype(int)
    df_inter['timestamp'] = df_inter['timestamp'].astype(int)
    df_inter['new_item_id'] = df_inter['new_item_id'].astype(int)

    df = pd.DataFrame(
        {'SessionID': df_inter['new_user_id'], 'Time': df_inter['timestamp'], 'ItemID': df_inter['new_item_id']})
    df.to_csv(save_path, index=False, header=True)


def removeShortSessions(data):
    # delete sessions of length < 1
    sessionLen = data.groupby('SessionID').size()  # group by sessionID and get size of each session
    data = data[np.in1d(data.SessionID, sessionLen[sessionLen > 1].index)]
    return data


def split():
    dataBefore = './datasets/Amazon_Toys_and_Games/TrustSR.csv'  # Path to Original Training Dataset "Clicks" File
    dataAfter = './datasets/Amazon_Toys_and_Games/'  # Path to Processed Dataset Folder
    dayTime = 36400000

    # Read Dataset in pandas Dataframe (Ignore Category Column)
    train = pd.read_csv(dataBefore, sep=',', header=None, usecols=[0, 1, 2], skiprows=1,
                        dtype={0: np.int32, 1: float, 2: float})
    train.columns = ['SessionID', 'Time', 'ItemID']  # Headers of dataframe

    train = removeShortSessions(train)
    # delete records of items which appeared less than 5 times
    itemLen = train.groupby('ItemID').size()  # groupby itemID and get size of each item
    train = train[np.in1d(train.ItemID, itemLen[itemLen > 2].index)]
    # remove sessions of less than 2 interactions again
    train = removeShortSessions(train)
    timeMax = train.Time.max()
    sessionMaxTime = train.groupby('SessionID').Time.max()
    sessionTrain = sessionMaxTime[
        sessionMaxTime < (timeMax - dayTime)].index  # training split is all sessions that ended before the last 2nd day
    sessionValid = sessionMaxTime[sessionMaxTime >= (
                timeMax - dayTime)].index  # validation split is all sessions that ended during the last 2nd day
    trainTR = train[np.in1d(train.SessionID, sessionTrain)]
    trainVD = train[np.in1d(train.SessionID, sessionValid)]
    # Delete records in validation split where items are not in training split
    trainVD = trainVD[np.in1d(trainVD.ItemID, trainTR.ItemID)]
    # Delete Sessions in testing split which are less than 2
    trainVD = removeShortSessions(trainVD)

    groups = trainVD.groupby('SessionID')
    group_list = list(groups)
    random.shuffle(group_list)
    test_split = int(0.5 * len(group_list))

    val_lines = group_list[:test_split]
    test_lines = group_list[test_split:]

    group_dataframes = [group_data for _, group_data in val_lines]
    val_data = pd.concat(group_dataframes)

    group_dataframes = [group_data for _, group_data in test_lines]
    test_data = pd.concat(group_dataframes)

    # Convert To CSV
    print('Training Set has', len(trainTR), 'Events, ', trainTR.SessionID.nunique(), 'Sessions, and',
          trainTR.ItemID.nunique(), 'Items\n\n')
    trainTR.to_csv(dataAfter + 'train.csv', sep=',', index=False)
    print('Validation Set has', len(val_data), 'Events, ', val_data.SessionID.nunique(), 'Sessions, and',
          val_data.ItemID.nunique(), 'Items\n\n')
    val_data.to_csv(dataAfter + 'valid.csv', sep=',', index=False)
    print('Testdata Set has', len(test_data), 'Events, ', test_data.SessionID.nunique(), 'Sessions, and',
          test_data.ItemID.nunique(), 'Items\n\n')
    test_data.to_csv(dataAfter + 'test.csv', sep=',', index=False)


