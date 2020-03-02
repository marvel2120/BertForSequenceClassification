import numpy as np
import os
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

def data_loader(data_file):
  data = []
  label = []
  with open(data_file) as f:
      for line in f.readlines():
        # sp = line.strip().split()
          lineLS = eval(line)
          tmpLS = lineLS[1].split()
          if "sarcasm" in tmpLS:
              continue
          if "sarcastic" in tmpLS:
              continue
          if "reposting" in tmpLS:
              continue
          if "<url>" in tmpLS:
              continue
          if "joke" in tmpLS:
              continue
          if "humour" in tmpLS:
              continue
          if "humor" in tmpLS:
              continue
          if "jokes" in tmpLS:
              continue
          if "irony" in tmpLS:
              continue
          if "ironic" in tmpLS:
              continue
          if "exgag" in tmpLS:
              continue

          data.append(lineLS[1])
          label.append(int(lineLS[-1]))
        # label.append(int(sp[-1]))
        # data.append(" ".join(sp[:-1]))
      label = np.array(label)
      return data, label

def data_process(data, max_len):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  input_ids = []
  attention_masks = []
  for sent in data:
    encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
    input_ids.append(encoded_sent)
  input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', value=0, truncating='post',padding='post')
  for sent in input_ids:
    att_mask = [int(token_id>0) for token_id in sent]
    attention_masks.append(att_mask)
  input_ids = np.array(input_ids)
  attention_masks = np.array(attention_masks)
  return input_ids, attention_masks

def batch_iter(x, y, att_mask, batch_size):
    """
    :param x: data
    :param y: label
    :param batch_size: how many samples in one single batch
    :return: a batch of data
    """
    data_len = len(x)
    num_batch = int(data_len / batch_size) if data_len % batch_size == 0 else int(data_len / batch_size)+1
    indices = np.random.permutation(np.arange(data_len))
    x = x[indices]
    y = y[indices]
    att_mask = att_mask[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = (i + 1) * batch_size
        yield x[start_id:end_id], y[start_id:end_id], att_mask[start_id:end_id]











