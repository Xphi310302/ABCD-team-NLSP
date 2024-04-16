from transformers import GenerationConfig



def read_txt(file_path) -> list:
    # Iniialize an empty list to store lines
    lines = []
    # Open the file with UTF-8 encoding and read it line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Append each line to the list
            lines.append(line.replace('\n', ''))
    return lines

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

def get_software_from_labels(sentence, labels):
    software_entities = []
    current_software = None
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if current_software is not None:
                software_entities.append(current_software)
            current_software = {'label': label[2:], 'text': [sentence[i]], 'position': i}
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

def format_number_to_text(label_ner, replacement_dict):
    # Split the input string based on ';;' to separate different segments
    segments = label_ner.split(";;")

    # Initialize an empty list to store the modified segments
    modified_segments = []

    # Iterate over the segments
    for segment in segments:
        # Split each segment based on '\t' to get the position and replace with corresponding value
        parts = segment.split("\t")
        if len(parts) == 0: continue
        # print(parts)
        modified_segment = segment
        # Extract position and replace with value from the dictionary
        for i in range(1, len(parts)):
            position = int(parts[i])
            value =  replacement_dict[position] # If position not in dictionary, use original value
            # Replace position with value in the segment
            modified_segment = modified_segment.replace(f"{position}", f"{value}")
        # Add the modified segment to the list
        modified_segments.append(modified_segment)


    # Join the modified segments back into a string
    output_string = ";;".join(modified_segments)
    return output_string

def format_text_to_number(label_ner, replacement_dict):
    def find_key_by_value(dictionary, target_value):
        for key, value in dictionary.items():
            if value == target_value:
                return key
    # Split the input string based on ';;' to separate different segments
    segments = label_ner.split(";;")

    # Initialize an empty list to store the modified segments
    modified_segments = []

    # Iterate over the segments
    for segment in segments:
        # Split each segment based on '\t' to get the position and replace with corresponding value
        parts = segment.split("\t")
        modified_segment = segment

        # Extract position and replace with value from the dictionary
        for i in range(1, len(parts)):
            # print(parts)
            value = parts[i]
            position = find_key_by_value(replacement_dict, value)# If position not in dictionary, use original value
            # Replace position with value in the segment
            modified_segment = modified_segment.replace(f"{value}", f"{position}")
        # Add the modified segment to the list
        modified_segments.append(modified_segment)

    # Join the modified segments back into a string
    output_string = ";;".join(modified_segments)
    return output_string

def format2text(format_text):
    # Split the input string based on ';;' to separate different segments
    segments = format_text.split(";;")
    # Initialize an empty list to store the modified segments
    modified_segments = []

    # Iterate over the segments
    for segment in segments:
        # Split each segment based on '\t' to get the values
        parts = segment.split("\t")
        if len(parts) > 2:
            # Reorder the values and create a new segment
            modified_segment = f"{parts[1]}\tis\t{parts[0]}\t{parts[2]}"
        else:
            modified_segment = segment
        # Add the modified segment to the list
        modified_segments.append(modified_segment)

    # Join the modified segments back into a string
    output_string = ";;".join(modified_segments)
    return output_string

def text2format(text):
    # Split the input string based on ';;' to separate different segments
    segments = text.split(";;")
    # Initialize an empty list to store the modified segments
    modified_segments = []
    # Iterate over the segments
    for segment in segments:

        # Split each segment based on '\t' to get the values
        parts = segment.split('\t')
        if len(parts) > 3:
            # print(len(parts))
            # Reorder the values and create a new segment
            # print(len(parts))
            modified_segment = f"{parts[2]}\t{parts[0]}\t{parts[3]}"
        else:
            modified_segment = segment
        # Add the modified segment to the list
        modified_segments.append(modified_segment)


    # Join the modified segments back into a string
    output_string = ";;".join(modified_segments)
    return output_string



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
    output = output[0].split("[/INST]")[-1].strip()
    return output

def parse_software_and_label(input_text):
    # Define the pattern for extracting software and labels
    parsed_output = {}
    if '\n' in input_text:
        list_of_software_with_label = input_text.split('\n')
    else:
        list_of_software_with_label = [input_text]
    for software_with_label in list_of_software_with_label:
        if '_' not in software_with_label:
            continue
        software = software_with_label.split('_')[0]
        label  = software_with_label.split('_')[1].strip()
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