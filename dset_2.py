import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
from tqdm import tqdm
import json

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

def get_string(dct):
    words = []
    for k, v in dct.items():
        words.append(k + ': ')
        if isinstance(v, dict):
            words += get_string(v)
        else:
            words += v.split()
    return words

def convert_to_graph(dct):
    words = get_string(dct)
    edge_index = [[i, i + 1] for i in range(len(words) - 1)]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = []
    for word in words:
        x.append(tokenizer.encode(word, return_tensors='pt')[0].to(dtype=torch.long))
    # pad the sequences
    max_len = max([len(xi) for xi in x])
    x = [F.pad(xi, (0, max_len - len(xi))) for xi in x]
    x = torch.stack(x)
    return Data(x=x, edge_index=edge_index.t().contiguous())

data_path = Path.cwd() / 'data/squad'
files = list(data_path.glob('*'))
res_data = []
with open(files[0], 'r') as f:
    data = json.load(f)
    question_count = 0
    prog_bar = tqdm(enumerate(data['data']), total=len(data['data']))
    for i, item in prog_bar:
        new_dc = {'title': item['title'], 'paragraphs': []}
        for j, paragraph in enumerate(item['paragraphs']):
            questions = []
            for k, qa in enumerate(paragraph['qas']):
                is_impossible = qa['is_impossible']
                question = qa['question']
                question_count += 1
                prog_bar.set_description(' Question {} Question Count {}'.format( question, question_count), refresh=True)
                answers = []
                plausible_answers = []
                if len(qa['answers']) == 0:
                    for ans in qa['plausible_answers']:
                        
                        answer = ans['text']
                        answer_start = ans['answer_start']
                        plausible_answers.append({'text': answer, 'answer_start': answer_start})
                else:
                    for ans in qa['answers']:
                        answer = ans['text']
                        answer_start = ans['answer_start']
                        answers.append({'text': answer, 'answer_start': answer_start})
                if len(answers) > 0:
                    questions.append({'question': question, 'answers': answers, 'is_impossible': is_impossible})
                else:
                    questions.append({'question': question, 'plausible_answers': plausible_answers, 'is_impossible': is_impossible})
            p = paragraph['context']
            
            new_dc['paragraphs'].append({'context': p, 'qas': questions})
        res_data.append(new_dc)
        
dset = []
for item in tqdm(res_data):
    dset_dct = {}
    outputs = {}
    for paragraph in item['paragraphs']:
        dset_dct['context'] = paragraph['context']
        
        for qa in paragraph['qas']:
            dset_dct['question'] = qa['question']
            if qa['is_impossible']:
                outputs['answers'] = qa['plausible_answers']
            else:
                outputs['answers'] = qa['answers']
            outputs['is_impossible'] = qa['is_impossible']
            dset.append((convert_to_graph(dset_dct), outputs))

import pickle
with open('data/squad/val_dset.pkl', 'wb') as f:
    pickle.dump(dset, f)
