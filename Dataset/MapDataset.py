from transformers import PreTrainedTokenizer
import torch

class MapDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer: PreTrainedTokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = [d['prompt'] for d in data]
        self.completions = [d['completion'] for d in data]

        # self.character_set = {'F', 'B', 'M', 'P', 'D', 'W'}
        # self.vocab_size = len(self.character_set)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = '<BOS>' + self.prompts[idx] + '<EOS>'
        completion = '<BOS>' + self.completions[idx] + '<EOS>'
        print(completion)

        prompt_encoding = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt", padding='max_length')
        completion_encoding = self.tokenizer(completion, add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt", padding='max_length')
        print(completion_encoding)
        # prompt = prompt_encoding['input_ids']
        # attention_mask = prompt_encoding['attention_mask']
        # completion = completion_encoding['input_ids']

        model_input = {}
        model_input['input_ids'] = torch.concatenate((prompt_encoding['input_ids'], completion_encoding['input_ids']), dim=1).squeeze()
        model_input['attention_mask'] = torch.concatenate((prompt_encoding['attention_mask'], completion_encoding['attention_mask']), dim=1).squeeze()
        model_input['labels'] = model_input['input_ids'].clone()
        for i, _ in enumerate(prompt_encoding['input_ids'][0]):
            model_input['labels'][i] = -100

        return {'input_ids': model_input['input_ids'], 'attention_mask': model_input['attention_mask'], 'labels': model_input['labels']}