import os
import torch
from transformers import FlaubertConfig, FlaubertTokenizer, FlaubertModel
from models import BertForSequenceClassification
from ut import *

class Jeux2BERT:
    def __init__(self, model_path):
        do_lower_case = False
        self.model_path = model_path
        self.config =  self.get_config() 
        self.lp_num_labels = getattr(self.config, "lp_num_labels")
        self.lp_label_list = ["0","1"]
        #setattr(self.config, "rp_num_labels", 132)
        self.rp_num_labels = getattr(self.config, "rp_num_labels") #132
        self.rp_label_list = get_relations(".")
        self.origin_entities = None
        self.entities = None
        
        self.max_seq_length = 128
        self.eval_batch_size = 50#360
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rel_tokenizer = FlaubertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        # Prepare model
        self.model = BertForSequenceClassification.from_pretrained(model_path, config=self.config)
        self.model = self.model.to(self.device)
        
    def get_config(self):
        return FlaubertConfig.from_pretrained(self.model_path)
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.rel_tokenizer
    
    def get_device(self):
        return self.device
    
    def lp(self, s, r, o):
        triple = [InputExample(guid=0, text_a= s.lower(), text_b= r, text_c= o.lower(), label="1")]

        label, score = inference(self.device, triple, self.lp_label_list, self.lp_num_labels, self.model,       
                                     self.rel_tokenizer, self.max_seq_length, lp_convert_examples_to_features, "lp")
    
        return label, score
    
    def rp(self, s, o):
        triple = [InputExample(guid=0, text_a= s.lower(), text_b= o.lower(), label="[r_associated]")]
    
        labels, scores = inference(self.device, triple, self.rp_label_list, self.rp_num_labels, self.model, 
                                       self.rel_tokenizer, self.max_seq_length, rp_convert_examples_to_features, "rp")
    
        return labels[::-1], scores[::-1]
    
    def triple_ranking(self, s, r, o, mode, entities):  
        #s, o = ("?", o) if mode=="head" else (s, "?") 
        triple = [InputExample(guid=0, text_a= s.lower(), text_b= r, text_c= o.lower(), label="1")]
        entities, scores = inference(self.device, triple, self.lp_label_list, self.lp_num_labels, self.model, 
                                     self.rel_tokenizer, self.max_seq_length, lp_convert_examples_to_features, mode, entities)
  
        return entities,scores


    def set_origin_entities(self, path):
        with open(os.path.join(".", path), encoding="latin-1", mode='r') as f:
            lines = f.readlines()
            entities= []
            for line in lines:
                entities.append(line.strip())
        return entities
    
    def set_entities(self, entities_arg):
        entities = []
        for e in entities_arg:
            entities.append(e.strip())
        return entities
    
    def take_decision(self, task, label, score):
        if(task=="lp"):
            #return "Ne sais pas"
            if (label==1):            
                if (score<  -0.35):
                    return "Ne sais pas"
                else:
                    return "Vrai"
            else:
                if (score< 0.0):
                    return "Ne sais pas"
                else:
                    return "Faux"
        
        elif(task=="rp"):
            relations = []
            for s,l in zip(score, label):
                if(s > - 5.5 and l not in ["[r_inhib]", "[r_lemma]", "[r_fem]", "[r_masc]" ]):
                    #if(l=="[r_lieu]" and "[r_isa]" in label and "[r_syn]" in label):
                    #    continue
                    if(l=="[r_isa]" and "[r_agent]" in label and "[r_patient]" in label):
                        continue
                    elif(l=="[r_isa]" and "[r_patient]" in label):
                        continue
                    elif(l=="[r_isa]" and "[r_agent]" in label):
                        continue
                    elif(l=="[r_isa]" and "[r_instr]" in label):
                        continue
                    elif(l=="[r_isa]" and "[r_lieu]" in label and "[r_syn]" not in label):
                        continue
                    else:
                        relations.append(l)
            
            if (relations is not None):
                return relations[:4]
            else:
                return "Ne sais pas"
        
