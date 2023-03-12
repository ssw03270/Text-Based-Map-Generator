import json
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler
from transformers import GPT2LMHeadModel, GPT2Tokenizer


from Dataset.MapDataset import MapDataset
from Utils import map_generator

start_time = time.time()
writer = SummaryWriter()

map_generator()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'})
# PAD: 배치 데이터 길이 맞추는 용더
# BOS: 문장의 시작을 알림
# EOS: 문장의 끝을 알림

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.resize_token_embeddings(len(tokenizer))

# train data 생성
with open("./Dataset/maps.jsonl", "r") as f:
    data = json.load(f)

train_dataset = MapDataset(data, tokenizer, max_length=200)
train_dataloader = DataLoader(train_dataset, batch_size=4)

num_epochs = 50

# Instantiate optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * num_epochs,
)

# Train the model
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        # Prepare batch data
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device)
        }
        # Forward pass
        loss = model(**inputs)[0]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print('epoch ' + str(epoch + 1) + '/' + str(num_epochs) + ': ' + str(loss.item()))
    writer.add_scalar("Loss", loss.item(), epoch + 1)

writer.close()
model.save_pretrained('./Model')
tokenizer.save_pretrained('./Tokenizer')