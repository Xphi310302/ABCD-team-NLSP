import re
import torch
from transformers import GenerationConfig

def preprocessing(sent):
    sent = re.sub(r"(https|http)?://\S+", " website ", sent).replace("website website", " website ").strip()
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent

def have_ner(sentence):
    return 1 if 'B' in sentence else 0

def get_software_from_labels(sentence, labels):
    software_entities = []

    current_software = None
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if current_software is not None:
                software_entities.append(current_software)
            current_software = {'label': label[2:], 'text': [sentence[i]]}
        elif label.startswith('I-'):
            if current_software is not None:
                current_software['text'].append(sentence[i])
        else:
            if current_software is not None:
                software_entities.append(current_software)
                current_software = None

    if current_software is not None:
        software_entities.append(current_software)

    return software_entities

def evaluate_model(model, tokenizer, sample):
    inputs = tokenizer(sample, return_tensors="pt").to('cuda')
    generation_config  = GenerationConfig(
    do_sample=True,
    max_new_tokens=256,
    top_k=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    penalty_alpha = 0.6,
    return_full_text=False,
    )

    outputs = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output = output[0].split("[/INST]")[1].strip()
    return output

def parse_software_and_label(input_text):
    # Define the pattern for extracting software and labels
    parsed_output = {}
    if '\n' in input_text:
        list_of_software_with_label = input_text.split('\n')
    else:
        list_of_software_with_label = [input_text]

    for software_with_label in list_of_software_with_label:
        software = software_with_label.split(':')[0]
        label  = software_with_label.split(':')[1].strip()
        parsed_output[software] = label

    return parsed_output

def tag_sentence(sentence, entity_dict):
    words = sentence.split()

    # Initialize a list to store the tags
    tags = ['O'] * len(words)

    # Iterate through the words
    i = 0
    while i < len(words):
        # Check if the word is part of any entity
        matched_entity = None
        for entity, category in entity_dict.items():
            entity_words = entity.split()
            if words[i:i+len(entity_words)] == entity_words:
                matched_entity = entity
                break

        if matched_entity:
            # Get the category from the dictionary
            category = entity_dict[matched_entity]

            # Create the tags based on the category
            tag_prefix = 'B-' + category if ' ' not in category else 'B-' + category.split()[0]
            tags[i] = tag_prefix
            for j in range(i + 1, i + len(matched_entity.split())):
                tags[j] = 'I-' + category.split()[0]
            i += len(matched_entity.split())
        else:
            i += 1

    return ' '.join(tags)