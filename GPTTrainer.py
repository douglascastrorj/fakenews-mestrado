import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import time
from datetime import datetime
from Dataset import Dataset


class GPTTrainer():
    
    def __init__(self, tokenizer, labels):
        self.tokenizer = tokenizer
        self.labels = labels

    def train(self, model, train_data, val_data, learning_rate, epochs, use_cuda):
        # set up GPU training (if using GPU)
        use_cuda = torch.cuda.is_available()
        print("Using Cuda : {}".format(use_cuda))
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
        device = torch.device("cuda" if use_cuda else "cpu")
        # set the seed for generating random numbers
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)
        # load train, validation and test data
        train, val = Dataset(train_data, self.tokenizer, self.labels), Dataset(val_data, self.tokenizer, self.labels)
        train_loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=2)
        # test_loader = _get_test_data_loader(args.batch_size, args.test_dir, **kwargs)
        # print logging info
        print(
            "Processes {}/{} ({:.0f}%) of train data".format(
                len(train_loader.sampler),
                len(train_loader.dataset),
                100.0 * len(train_loader.sampler) / len(train_loader.dataset),
            )
        )
        print(
            "Processes {}/{} ({:.0f}%) of val data".format(
                len(val_loader.sampler),
                len(val_loader.dataset),
                100.0 * len(val_loader.sampler) / len(val_loader.dataset),
            )
        )
        print(
            "Processes {}/{} ({:.0f}%) of test data".format(
                len(test_loader.sampler),
                len(test_loader.dataset),
                100.0 * len(test_loader.sampler) / len(test_loader.dataset),
            )
        )
        # initialize model and parameters
        # model = SimpleGPT2SequenceClassifier(hidden_size=args.hidden_size, num_classes=5, max_seq_len=args.max_seq_len, gpt_model_name="gpt2").to(device)
        EPOCHS = epochs
        LR = learning_rate
        # use cross-entropy as the loss function
        criterion = nn.CrossEntropyLoss()
        # use Adam as the optimizer
        optimizer = Adam(model.parameters(), lr=LR)
        # enable GPU training (if using GPU)
        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
        # training loop
        for epoch_num in range(EPOCHS):
            total_acc_train = 0
            total_loss_train = 0
            
            for train_input, train_label in tqdm(train_loader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input["input_ids"].squeeze(1).to(device)
                
                model.zero_grad()
                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1)==train_label).sum().item()
                total_acc_train += acc
                batch_loss.backward()
                optimizer.step()
                
            total_acc_val = 0
            total_loss_val = 0
            
            # validate model on validation data
            with torch.no_grad():
                for val_input, val_label in val_loader:
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    
                    output = model(input_id, mask)
                    
                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1)==val_label).sum().item()
                    total_acc_val += acc
                    
                logger.info(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_loader): .3f} \
                | Train Accuracy: {total_acc_train / len(train_loader.dataset): .3f} \
                | Val Loss: {total_loss_val / len(val_loader.dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_loader.dataset): .3f}")
        # evaluate model performance on unseen data
        # test(model, test_loader, device)
        
        # save model
        # save_model(model, args.model_dir)