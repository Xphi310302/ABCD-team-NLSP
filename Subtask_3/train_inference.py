import torch
import pandas as pd
from model import initialize_model
from utilis import *
from prompt import create_format
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator
from training import train_model
# hyperparameters
lr = 2e-4
num_epochs = 12
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'models/llama_model'



# Read text data
file_path = 'subtask3/subtask3_train.data.txt'  # Replace with the path to your text file
sentence = read_txt(file_path)
file_path = 'subtask3/subtask3_train.labels.txt'
label = read_txt(file_path)
file_path = 'subtask3/subtask3_train.info.txt'
info = read_txt(file_path)
data = pd.DataFrame({'sentence': sentence, 'label': label, 'info': info})
data['have_ner'] = data['info'].apply(lambda x: have_ner(x))
data = data.rename(columns = {'label': 'label_ner'})

# preprocessing 
examples_list = []
mapping_list = []
template = """Software and infomation mention:
{info_list}"""
for i in data.index:
    # print(i)
    sentence = data.loc[i,'sentence']
    info = data.loc[i,'info']
    info_entities = get_software_from_labels(sentence.split(' '), info.split(' '))

    info_list = []
    mapping = {}
    for entity in info_entities:
        position = entity['position']
        info_name =  ' '.join(entity['text'])  + f'(position={position})'
        info_list.append(f"- {info_name}>>{entity['label']}")
        # save mapping
        mapping[position] = ' '.join(entity['text'])
    info_list_str = '\n'.join(info_list)
    examples_list.append(template.format(info_list = info_list_str))

    # save mapping
    mapping_list.append(mapping)
data['example'] = examples_list
data['mapping'] = mapping_list
data = data[data['have_ner'] == 1]

# Apply the function to a column using apply
data['label'] = data.apply(lambda row: format_number_to_text(row['label_ner'], row['mapping']), axis=1)
data['label'] = data['label'].apply(lambda col: format2text(col))
data = create_format(data)

# create train test
test_df = data[data['have_ner'] == 1][:50]
train_df = data[data['have_ner'] == 1]
train_df = train_df.drop(columns = ['have_ner'])[['text','label', 'sentence']]
test_df = test_df.drop(columns = ['have_ner'])[['text','label', 'sentence']]

# initialize model 
model, tokenizer = initialize_model(model_path)

# create dataset
tds = Dataset.from_pandas(train_df)
teds = Dataset.from_pandas(test_df)
dataset = DatasetDict()
dataset['train'] = tds
dataset['test'] = teds

# add pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
max_length = max([len(tokenizer(review)["input_ids"]) for review in train_df["text"].tolist()])

# prerocessing for training
def preprocess_function(examples):
    batch_size = len(examples["text"])
    inputs = [item + " " for item in examples["text"]]
    targets = examples["label"]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# train model
model = train_model(model, tokenizer, train_dataloader, eval_dataloader, num_epochs, lr)

# Prediction
# Read test data
file_path = 'subtask3/subtask3_test.data.txt'  # Replace with the path to your text file
sentence = read_txt(file_path)
file_path = 'subtask3/subtask3_test.info.txt'
info = read_txt(file_path)
test_df = pd.DataFrame({'sentence': sentence, 'info': info})
# data['label'] = entities_with_label_list
examples_list = []
mapping_list = []
template = """Software and infomation mention:
{info_list}"""
for i in test_df.index:
    # print(i)
    sentence = test_df.loc[i,'sentence']
    info = test_df.loc[i,'info']
    info_entities = get_software_from_labels(sentence.split(' '), info.split(' '))
    info_list = []
    mapping = {}
    for entity in info_entities:
        position = entity['position']
        info_name =  ' '.join(entity['text'])  + f'(position={position})'
        info_list.append(f"- {info_name}>>{entity['label']}")
        # save mapping
        mapping[position] = ' '.join(entity['text'])
    info_list_str = '\n'.join(info_list)
    examples_list.append(template.format(info_list = info_list_str))

    # save mapping
    mapping_list.append(mapping)
test_df['example'] = examples_list
test_df['mapping'] = mapping_list
test_df = create_format(test_df)

# predict
model.config.use_cache = True
for i in tqdm(test_df.index):
    sentence = test_df.loc[i, 'sentence']
    sample = test_df.loc[i, 'text']
    replacement_dict = test_df.loc[i, 'mapping']
    pred_text = evaluate_model(model, tokenizer, sample)
    output = format_text_to_number(text2format(pred_text), replacement_dict)
    with open('predictions.txt', 'a', encoding='utf-8') as output_file:
        output_file.write(output + '\n')