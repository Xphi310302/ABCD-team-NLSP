import pandas as pd
from sklearn.metrics import *
import torch
from prompt import create_format
from model import initialize_model
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator
from training import train_model
from utilis import *

#max_length_input=196
lr = 2e-4
num_epochs = 12
batch_size = 4
model_path = 'models/llama_model'
test_data = 'Test.csv'
train_data = 'train.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Read train dataset
data = pd.read_csv(train_data)
data['sentence'] = data['sentence'].apply(lambda x: preprocessing(x))
data['have_ner'] = data['label'].apply(lambda x: have_ner(x))
data = data.rename(columns = {'label': 'label_ner'})

# preprocess data
entities_with_label_list = []
for i in data.index:
    sentence = data.loc[i,'sentence']
    label = data.loc[i,'label_ner']
    # Get software positions
    software_entities1  = get_software_from_labels(sentence.split(' '), label.split(' '))
    # Print the results
    entities_with_label = ''
    for entity in software_entities1:
        software_name = ' '.join(entity['text'])
        entities_with_label += f"{software_name}:{entity['label']}\n"
        # print(entities_with_label)
    entities_with_label_list.append(entities_with_label)
data['label'] = entities_with_label_list
data = create_format(data)

# Create train, test 
test_df = data[data['have_ner'] == 1][2000:2050]
train_df = data[data['have_ner'] == 1]
train_df = train_df.drop(columns = ['have_ner'])
test_df = test_df.drop(columns = ['have_ner'])
# Create dataset
tds = Dataset.from_pandas(train_df)
teds = Dataset.from_pandas(test_df)
dataset = DatasetDict()
dataset['train'] = tds
dataset['test'] = teds

# initilize model
model, tokenizer = initialize_model(model_path)
# add pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
max_length = max([len(tokenizer(review)["input_ids"]) for review in train_df["text"].tolist()])

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

# Initialize dataloader
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# Train model
model = train_model(model, tokenizer, train_dataloader, eval_dataloader, num_epochs, lr)

# Prediction
test_df = pd.read_csv(test_data)
test_df = test_df.rename(columns={"text": "sentence"})
test_df = create_format(test_df).drop([8180], axis = 0)
model.config.use_cache = True

for i in tqdm(test_df.index):
    sentence = test_df.loc[i, 'sentence']
    if test_df.loc[i, 'label'] == 1:
        sample = test_df.loc[i, 'text']
        pred_text = evaluate_model(model, tokenizer, sample)
        # print(pred_text)
        parsed_prediction = parse_software_and_label(pred_text)
        tags = tag_sentence(sentence, parsed_prediction)
    else:
        parsed_prediction = {}
        tags = tag_sentence(sentence, parsed_prediction)
    with open('predictions.txt', 'a', encoding='utf-8') as output_file:
        output_file.write(tags + '\n')