import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
import requests
from PIL import Image
import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pickle


dataset_path = './datasets/Amazon_Toys_and_Games'
inter_data = 'user_review.csv'
item_data = 'item_review.csv'
save_data = 'my_data_meta.csv'
inter_path = os.path.join(dataset_path, f'{inter_data}')
item_path = os.path.join(dataset_path,f'{item_data}')
save_path = os.path.join(dataset_path,f'{save_data}')

# get interaction_sequence and correspond review sequence
inter_usecols = ['user_id','item_id','timestamp','reviewText']
inter_dtype = {'user_id':str,'item_id':str,'timestamp':int,'reviewText':str}

# get item reviews and image
item_usecols = ['item_id','reviewText','imageURLHighRes','description']
item_dtype = {'item_id':str,'reviewText':str,'imageURLHighRes':str,'description':str}



def create_Dataframe():

    inter_list = []
    with open("./datasets/Amazon_Toys_and_Games/Toys_and_Games_5.json", 'r') as freview:
        for line in tqdm(freview):
            line = line.rstrip()
            if line.endswith('}'):
                json_data = json.loads(line)
                if "reviewText" in json_data.keys():
                    json_data["reviewText"] = json_data["reviewText"].replace('\n', '')
                    json_data["reviewText"] = json_data["reviewText"].replace("\"", "")
                    # d["reviewText"] = d["reviewText"].replace("\\", "")
                    inter_list.append([json_data["reviewerID"], json_data["asin"], json_data["overall"], json_data["unixReviewTime"], json_data["reviewText"]])


    df_inter = pd.DataFrame(inter_list, columns=['user_id', 'item_id', 'rate', 'timestamp', 'reviewText'])  # 原始df
    df_inter = df_inter.drop_duplicates()

    item_list = []

    with open("./datasets/Amazon_Toys_and_Games/meta_Toys_and_Games.json", 'r') as item_f:
        for line in tqdm(item_f):
            line = line.rstrip()
            if line.endswith('}'):
                try:
                    json_data = json.loads(line)
                    if json_data["imageURLHighRes"] and json_data["description"]:
                        json_data["description"][0] = json_data["description"][0].replace('\n', '')
                        json_data["description"][0] = json_data["description"][0].replace("\"", "")
                        image_list = json_data["imageURLHighRes"][0].split(",")
                        image = image_list[0]
                        item_list.append([json_data["asin"], json_data["description"][0], image])
                except json.JSONDecodeError:
                    pass

    df_item = pd.DataFrame(item_list, columns=['item_id', 'description', "imageURLHighRes"])

    df_item = df_item.drop_duplicates()
    return df_inter,df_item


def choose_dataframe():

    inter_df,item_df = create_Dataframe()

    item_id_unique_df = inter_df[['item_id']].drop_duplicates()
    new_item_df = pd.merge(item_id_unique_df[["item_id"]],item_df[['item_id', 'description', "imageURLHighRes"]],on='item_id',how='inner')
    

    item_review_df = inter_df.groupby("item_id")["reviewText"].apply(lambda x: x.str.cat(sep='\",\"')).reset_index()

    item_review_result = pd.merge(item_review_df[['item_id', 'reviewText']],
                                  new_item_df[['item_id', 'description', "imageURLHighRes"]], on='item_id', how='inner')

    item_id_unique_df1 = item_review_result[['item_id']].drop_duplicates()
    new_inter_df = pd.merge(item_id_unique_df1[["item_id"]],inter_df[['user_id', 'item_id', 'rate', 'timestamp', 'reviewText']],on='item_id',how='inner')

    new_inter_df.to_csv('./datasets/Amazon_Toys_and_Games/user_review.csv', index=False)
    item_review_result.to_csv('./datasets/Amazon_Toys_and_Games/item_review.csv', index=False)




def denoise():
    

    with open(inter_path, 'r') as f:  # loda interaction data
        df_inter = pd.read_csv(inter_path, delimiter=',', usecols=inter_usecols, dtype=inter_dtype)
        df_inter['new_user_id'], user_map = pd.factorize(df_inter['user_id'])
        df_inter['new_item_id'], item_map = pd.factorize(df_inter['item_id'])

    with open(item_path,'r') as f_i:
        df_item = pd.read_csv(item_path, delimiter=',', usecols=item_usecols, dtype=item_dtype)
        item_id = df_item['item_id'].tolist()
        df_item['new_item_id'] = pd.Index(item_map).get_indexer(item_id)


    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    user_review_seq = []
    item_review_seq = []
    index_seq = [i for i in range(len(df_inter))]
    r_score = []
    image_seq = []
    des_seq = []
    review_nums = 3
    pbar = tqdm(range(len(user_map)))
    t=0
    for user_id, group in df_inter.groupby('new_user_id'):
        item_id = group['new_item_id'].tolist()
        user_review = group['reviewText'].tolist()
        if len(user_review)>1:
            str = ''.join(user_review)
            for i in range(len(user_review)):
                user_review_seq.append(str)
            user_review_emb = model.encode(str, convert_to_tensor=True)

        else:
            user_review_seq.append(user_review[0])
            user_review_emb = model.encode(user_review[0],convert_to_tensor=True)



        item = pd.DataFrame()
        for id in item_id:
            temp_df = df_item.loc[df_item['new_item_id'] == id]
            item = pd.concat([item, temp_df])
        item_review = item['reviewText'].tolist()

        if len(item_review)>1:
            for index, i in enumerate(item_review):
                r = tuple(i.split('","'))
                item_review_emb = model.encode(r,convert_to_tensor=True)
                score = util.pytorch_cos_sim(item_review_emb,user_review_emb)
                score = score.cpu()
                sort = sorted(zip(score.tolist(),r),reverse=True)
                r = [x[1] for x in sort]
                item_review_seq.append(r[:review_nums])
                r_score.append(score)
                image_seq.append(item['imageURLHighRes'].tolist()[index])
                des_seq.append(item['description'].tolist()[index])

        else:
            item_review[0] = tuple(item_review[0].split('","'))
            item_review_emb = model.encode(item_review[0], convert_to_tensor=True)
            score = util.pytorch_cos_sim(item_review_emb,user_review_emb)
            score = score.cpu()
            sort = sorted(zip(score.tolist(),item_review[0]),reverse=True)
            item_review[0] = [x[1] for x in sort]
            r_score.append(score)
            item_review_seq.append(item_review[0][:review_nums])
            image_seq.append(item['imageURLHighRes'].tolist()[0])
            des_seq.append(item['description'].tolist()[0])

        t = t+1
        if len(user_review_seq)!=len(image_seq):
            with open('var.pickle', 'wb') as f:
                pickle.dump((item_id), f)
        pbar.update(1)

    l = len(r_score)
    inter_temp = df_inter.iloc[0:l]
    with open('image_seq.pickle', 'wb') as f:
        pickle.dump(inter_temp['user_id'], f)
    df = pd.DataFrame({'index':index_seq, 'u_name':inter_temp['user_id'],'i_name':inter_temp['item_id'],'u_id': inter_temp['new_user_id'], 'i_id': inter_temp['new_item_id'], 'score':r_score,
                    'u_review':user_review_seq, 'i_review':item_review_seq,'image':image_seq, 'description':des_seq})
    df.to_csv(save_path,index=False)



def download_image():
    item_data = pd.read_csv("./datasets/Amazon_Toys_and_Games/item_review.csv")
    for row in item_data.itertuples():
        item_name = row[1]
        if os.path.isfile('./datasets/Amazon_Toys_and_Games/image/{}.jpg'.format(item_name)):
            print("{} image has been download".format(item_name))
            continue
        else:
            try:
                url = row[4]
                response = requests.get(url, stream=True).raw
                image = Image.open(response)
                image.save('./datasets/Amazon_Toys_and_Games/image/{}.jpg'.format(item_name),'JPEG')
                print(item_name)
                response.close()
            except (OSError, NameError):
                image = Image.open('./datasets/Amazon_Toys_and_Games/image/0020232233.jpg')
                image.save('./datasets/Amazon_Toys_and_Games/image/{}.jpg'.format(item_name),'JPEG')
                print(item_name)

    print("finish download image")

