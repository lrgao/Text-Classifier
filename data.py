import json
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


class CLSDataLoader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        
        # super(WebNLGDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
        #                                 num_workers=args.num_workers)
        
        # TODO by lrgao
        super(CLSDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                               num_workers=0)

class CLSDataset(Dataset):
    def __init__(self, args, data_path, tokenizer, mode, attribute):
        self.data_path = data_path
        self.tokenizer = tokenizer
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

        print("Total samples = {}".format(len(self.data)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode
        self.metric = "ACC"
        self.attribute = attribute

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.labels, self.label2ids, self.id2labels = self.read_label(args.label_file)

    def __len__(self):
        return len(self.data)
    
    def read_label(self, label_file):
        labels = []
        with open(label_file, 'r') as f:
            label2ids= json.load(f)

        labels = list(set(list(label2ids.keys())))

        id2labels = {}
        for key,value in label2ids.items():
            id2labels[value]=key
        assert len(labels) == len(label2ids) == len(id2labels)

        return labels,label2ids,id2labels
    
    def get_input_ids(self, input_string):
        max_len = self.args.max_input_length

        if len(input_string) > max_len-2:
            input_string = input_string[:max_len-2]
            
        input_ids =self.tokenizer.encode(" {}".format(input_string), max_length=max_len, truncation=True, add_special_tokens=True)
        attn_mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))
        # padding
        input_ids += [self.tokenizer.pad_token_id] * (max_len - len(input_ids))
        assert len(input_ids) == len(attn_mask) == max_len
        return input_ids, attn_mask
    
    def __getitem__(self, idx):
        entry = self.data[idx]

        input_string = entry['description']
        input_ids, input_mask = self.get_input_ids(input_string)
        input_labels = entry[self.attribute]
        
        # transfer input labels into ids 
        label_ids = [self.label2ids[input_labels]]

        input_ids = torch.LongTensor(input_ids)
        input_mask = torch.LongTensor(input_mask)
        label_ids = torch.LongTensor(label_ids)

        return input_ids, input_mask, label_ids
