import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 모델 경로 설정
model_path = './Model'
tokenizer_path = './Tokenizer'

# tokenizer와 model 불러오기
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
tokenizer.add_special_tokens({'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'})
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True).to(device)
model.resize_token_embeddings(len(tokenizer))

# 입력 텍스트 설정
text = "The door is located on the bottom. Some blocks exist. No monsters exist. No elements exist."

# 입력 텍스트 토큰화
prompt_encoding = tokenizer(text, add_special_tokens=True, truncation=True, max_length=200, return_tensors="pt", padding='max_length')

prompt = prompt_encoding['input_ids']
attention_mask = prompt_encoding['attention_mask']

# 모델에 입력값 전달
with torch.no_grad():
    outputs = model.generate(input_ids=prompt.to(device), attention_mask=attention_mask.to(device), max_length=400, pad_token_id=tokenizer.eos_token_id)

decoded_outputs = tokenizer.decode(outputs[0].detach().cpu().squeeze(), skip_special_tokens=True)
print(decoded_outputs)
decoded_outputs = decoded_outputs.replace(text, '')
for i in range(0, 176, 16):
    print(decoded_outputs[i:i+11])