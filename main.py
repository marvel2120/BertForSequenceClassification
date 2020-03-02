import torch
import utils
import time
import datetime
import random
from config import Config
import numpy as np
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, \
    confusion_matrix

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(" Using: ", torch.cuda.get_device_name(0))
config = Config()

seed_val = 123
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=1e-3)

max_len = 75

train_data, train_label = utils.data_loader(config.train_path)
valid_data, valid_label = utils.data_loader(config.valid_path)
test_data, test_label = utils.data_loader(config.test_path)

total_steps = (len(train_data)/config.batch_size)*config.epoch
schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps=total_steps)

train_ids, train_attention_masks = utils.data_process(train_data, max_len)
valid_ids, valid_attention_masks = utils.data_process(valid_data, max_len)
test_ids, test_attention_masks = utils.data_process(test_data, max_len)

train_losses = []
valid_losses = []

def format_time(elapsed):
  elapsed_rounded = int(round(elapsed))
  return str(datetime.timedelta(seconds = elapsed_rounded))

def evaluate(y_pred, y_true):
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1 score: ", f1_score(y_true, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

def validation():
    model.eval()
    batch_valid = utils.batch_iter(valid_ids, valid_label, valid_attention_masks, config.batch_size)
    eval_acc = 0
    eval_loss = 0
    count=0
    for x_batch, y_batch, attn_mask in batch_valid:
        x_batch = torch.from_numpy(x_batch).type(torch.LongTensor).to(device)
        y_batch = torch.from_numpy(y_batch).type(torch.LongTensor).to(device)
        attn_mask = torch.from_numpy(attn_mask).type(torch.LongTensor).to(device)
        with torch.no_grad():
            outputs = model(x_batch, token_type_ids=None, attention_mask=attn_mask,labels=y_batch)
        loss = outputs[0]
        num_correct = (torch.max(outputs[1], 1)[1] == y_batch.data).sum()
        acc = (100 * num_correct )/ len(x_batch)
        eval_loss += loss.item()
        eval_acc += acc.item()
        count+=1
    eval_loss = eval_loss / count
    eval_acc = eval_acc / count
    torch.save(model.state_dict(), config.model_save_path)
    return eval_loss, eval_acc

def train():
  for epoch in range(config.epoch):
    model.train()
    print('Epoch: {0:02}'.format(epoch+1))
    t0 = time.time()
    total_epoch_loss = 0
    total_epoch_acc = 0
    batch_train = utils.batch_iter(train_ids, train_label,train_attention_masks, config.batch_size)
    steps = 1
    for x_batch, y_batch, attn_mask in batch_train:
      x_batch = torch.from_numpy(x_batch).type(torch.LongTensor).to(device)
      y_batch = torch.from_numpy(y_batch).type(torch.LongTensor).to(device)
      attn_mask = torch.from_numpy(attn_mask).type(torch.LongTensor).to(device)
      model.zero_grad()
      outputs = model(x_batch, token_type_ids=None, attention_mask=attn_mask,labels=y_batch)
      loss = outputs[0]
      total_epoch_loss+=loss.item()
      num_correct = (torch.max(outputs[1], 1)[1] == y_batch.data).sum()
      acc = (100 * num_correct )/ len(x_batch)
      total_epoch_acc+=acc.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
      optimizer.step()
      schedule.step()
      if steps % 100 == 0:
        print("batch:", steps)
        print('Training Loss: {0:.4f}'.format(loss.item()), 'Training Accuracy: {0: .2f}%'.format(acc.item()))
        print("Time elaspsed: ", format_time(time.time()-t0))
      steps+=1
    print('Epoch: {0:02}'.format(epoch + 1))
    print('Train Loss: {0:.3f}'.format(total_epoch_loss / steps),
              'Train Acc: {0:.3f}%'.format(total_epoch_acc / steps))
    eval_loss, eval_acc = validation()
    print('Validation Loss: {0:.3f}'.format(eval_loss), 'Validation Acc: {0:.3f}%'.format(eval_acc))
    train_losses.append(total_epoch_loss / steps)
    valid_losses.append(eval_loss)
    print("\n")

def test():
    batch_test = utils.batch_iter(test_ids, test_label, test_attention_masks, batch_size=config.batch_size)
    y_true = []
    y_pred = []
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()

    for x_batch, y_batch, attn_mask in batch_test:
        x_batch = torch.from_numpy(x_batch).type(torch.LongTensor).to(device)
        y_batch = torch.from_numpy(y_batch).type(torch.LongTensor).to(device)
        attn_mask = torch.from_numpy(attn_mask).type(torch.LongTensor).to(device)
        outputs = model(x_batch, token_type_ids=None, attention_mask=attn_mask, labels=y_batch)
        prediction = outputs[1]
        y_pred_batch = torch.max(prediction, 1)[1]
        y_true.extend(y_batch.tolist())
        y_pred.extend(y_pred_batch.tolist())
    evaluate(y_pred, y_true)

train()
test()
















