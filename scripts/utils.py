import os
import torch
from torch.utils.data import Dataset

# For fixing seeds
def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# A custom Dataset class for preparing dataset
class seq2seqDataset(Dataset):

    def __init__(self, data_df, tokenizer, max_source_len, max_target_len):

        self.tokenizer = tokenizer
        self.data = data_df
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source = self.tokenizer(
            self.data.source.to_list(),
            padding='longest',
            truncation=True,
            max_length=self.max_source_len,
            return_tensors="pt"
        )

        with self.tokenizer.as_target_tokenizer():
            self.target = self.tokenizer(
                self.data.target.to_list(),
                padding='longest',
                truncation=True,
                max_length=self.max_target_len,
                return_tensors="pt"
            )
        
    def __len__(self):

        assert len(self.source['input_ids']) == len(self.target['input_ids'])

        return len(self.source['input_ids'])

    def __getitem__(self, index):

        labels = [(label if label != self.tokenizer.pad_token_id else -100) for label in self.target['input_ids'][index]]
            
        return {
            'input_ids': self.source['input_ids'][index],
            'attention_mask': self.source['attention_mask'][index],
            'labels': torch.tensor(labels)
        }
