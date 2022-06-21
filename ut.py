import os
import torch
from torch.utils.data import (DataLoader, SequentialSampler,TensorDataset)
import numpy as np
import logging
import copy
from tqdm import tqdm, trange
logger = logging.getLogger(__name__)
eval_batch_size = 360

def get_relations(data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations_s.txt"), encoding="latin-1", mode='r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append("["+line.strip()+"]")
        return relations
    
    
def get_entities(data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities_s.txt"), encoding="latin-1", mode='r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations
    

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()
            
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, text_b2=None, text_c2=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_b2 = text_b2
        self.text_c2 = text_c2
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, tokens=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.tokens = tokens      


def ranking(entities, slot, example):
   
    examples = [example[0]]
    #print(example[0])
    rank_list = [[example[0].text_a, example[0].text_b, example[0].text_c ]]
    #print(rank_list)
    if(slot=="head"):
        for e in list(set(entities)):#[:100] :
            tmp = copy.deepcopy(example[0])
            tmp.text_a = e
            examples.append(tmp)
            #print(tmp.text_a, tmp.text_b, tmp.text_c)
            rank_list.append([tmp.text_a, tmp.text_b, tmp.text_c ])
            
    elif(slot=="tail"):
        for e in list(set(entities)):#[:100] :
            tmp =copy.deepcopy(example[0])
            tmp.text_c = e
            examples.append(tmp)
            #print(tmp.text_a, tmp.text_b, tmp.text_c)
            rank_list.append([tmp.text_a, tmp.text_b, tmp.text_c ])

            
    return examples,rank_list
        
        
def lp_convert_examples_to_features(examples, label_list, max_seq_length,tokenizer, print_info = True):
    """"""
    """Loads a data file into a list of `InputBatch`s for the Link Prediction task.
       ex) the triple <plant tissue, _hypernym, plant structure> should be converted to
       "[CLS] plant tissue, the tissue of a plant [SEP] hypernym [SEP] plant structure, \\
        any part of a plant or fungus [SEP]"
    """

    label_map = {label : i for i, label in enumerate(label_list)}
    features =  []
    for (ex_index, example) in enumerate(examples):
        
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        tokens_c = None
        tokens_b2 = None
        tokens_c2 = None
        #print(example.text_a, example.text_b, example.text_c)

        if example.text_b2 and example.text_c2:

            tokens_b2 = rel_tokenizer.tokenize(example.text_b2)
            tokens_c2 = tokenizer.tokenize(example.text_c2)
            _truncate_seq_triple(tokens_a, tokens_b2, tokens_c2, max_seq_length - 4)
        
        if example.text_b and example.text_c:
            tokens_b = [example.text_b]  
            tokens_c = tokenizer.tokenize(example.text_c)
            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b2: 
            tokens += tokens_b + ["[SEP]"]
            tokens += tokens_c + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            tokens += tokens_b2 + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b2) + 1)
            tokens += tokens_c2 + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c2) + 1)
        else:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)     
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #print(tokens)
        #print(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        #print(input_mask)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        #print(segment_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 0 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              tokens=[example.text_a,example.text_b,example.text_c]))
    
    return features

def rp_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(np.random.permutation(examples)):#[:1000]
        
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              tokens=[example.text_a,example.text_b]))
        
    return features

def inference( device, eval_examples, label_list, num_labels, model, rel_tokenizer,max_seq_length,
             convert_examples_to_features, task, entities = None):
        
        
    if(task=="head" or task=="tail"):
        #eval_examples,rank_list = ranking(get_entities("."), task, eval_examples)
        eval_examples,rank_list = ranking(entities, task, eval_examples)
        eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, rel_tokenizer)
    if(task=="lp"):
        eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, rel_tokenizer)
    else:
        eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, rel_tokenizer)
    
    logger.info("***** Running inference *****")
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_tokens = [f.tokens for f in eval_features]

    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []
    i=0
    origin = task
    if(task=="head" or task=="tail"):
        task="lp"
        
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        #tokens_id = all_tokens[i]

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, task=task)
            logits = outputs#[0]


        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    
    preds = preds[0]
    examp = np.array([])
    
    if(origin=="lp"):
        return np.argmax(preds, axis=1), preds[:, np.argmax(preds, axis=1)] 
        #return np.argmax(preds, axis=1), preds
    
    if(origin=="rp"):
        #print()
        return list(zip(*reversed([(label_list[e],d) for e,d in zip(np.argsort(preds[0])[::-1][:4], np.sort(preds[0])[::-1][:4])])))
    
    
    if(origin=="head" or origin=="tail"):          
        # get the dimension corresponding to current label 1
        rel_values = preds[:, 1] #preds[:, all_label_ids[0]]
        
        rel_values = torch.tensor(rel_values)
        argvalues1, argsort1 = torch.sort(rel_values, descending=True)
        argvalues1 = argvalues1.cpu().numpy()
        argsort1 = argsort1.cpu().numpy()         
        scores, entities = [], []
        for j in range(20):
            __idx = argsort1[j]
            if(origin=="head"):
                e, _, _ = rank_list[__idx]
            else:
                _, _, e = rank_list[__idx]
            scores.append(str(round(argvalues1[j], 4)))
            entities.append(e)
        
        return entities,scores
