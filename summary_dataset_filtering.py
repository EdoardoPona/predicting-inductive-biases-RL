import json
import os
import tempfile

# utilities to modify and analyze the huge summary dataset

filename='/home/eliasschmied/Downloads/tldr-challenge-dataset/tldr-training-data.jsonl'
outputfilename='/home/eliasschmied/ASH_code/predicting-inductive-biases-RL/tldr-filtered-training-data.jsonl'

def filter_function(obj):
    return len(obj.get("content").split(' ')) < 130

def filter_data():
    with open(filename, 'r') as input_file, open(outputfilename, 'w') as output_file:
        for line in input_file:
            obj = json.loads(line) 
            if filter_function(obj):
                output_file.write(json.dumps(obj) + '\n')

def check_filtered_dataset():
    with open(outputfilename, 'r') as input_file:
        count = 0
        for line in input_file:
            if count > 3:
                break
            obj = json.loads(line)
            count += 1
            print(line)
            if not filter_function(obj):
                print(count, line)
                break
        print(count)

def check_filtered_for_title():
    with open(outputfilename, 'r') as input_file:
        count = 0
        numberofmisses = 0
        for line in input_file:
            if count > 50 or numberofmisses > 20:
                break
            obj = json.loads(line)
            count += 1
            if "title" not in obj:
                print(count, 'no title found!', line)
        print(count)

def make_title_fields():
    temp_fd, temp_path = tempfile.mkstemp()

    with open(outputfilename, "r") as source_file, os.fdopen(temp_fd, 'w') as temp_file:
        for line in source_file:
            entry = json.loads(line)
            entry.setdefault("title", "")  # If "title" doesn't exist, set it to an empty string
            temp_file.write(json.dumps(entry) + "\n")

    # Replace the original file with the modified temporary file
    os.replace(temp_path, outputfilename)

check_filtered_for_title()