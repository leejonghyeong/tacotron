from utils import TacotronDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np

def run_train(n_epochs,
              model,
              optimizer,
              device,
              train_dataloader,
              dev_dataloader):
  
  #run_train: compute loss and update gradient
  for epoch in range(n_epochs):
    model.train()
    train_loss_stack = []
    with tqdm(total = len(train_dataloader), desc = f"Train {epoch}") as pbar:
      for train_batch in train_dataloader:
        text, mel_targets, lin_targets = (tensor.to(device) for tensor in train_batch)
        mel_outputs, lin_outputs, _ = model(text, mel_targets)

        optimizer.zero_grad()
        #L1 loss
        mel_loss = abs(mel_outputs - mel_targets)
        lin_loss = abs(lin_outputs - lin_targets)
        loss = mel_loss + lin_loss
        loss.backward()
        optimizer.step()

        train_loss_stack.append(loss.item())
        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(train_loss_stack):.3f})")
    
    print(f"[TRAIN] EP: {epoch} Avg: {np.mean(train_loss_stack):.4f}")

    model.eval()
    dev_loss_stack = []
    with tqdm(total = len(dev_dataloader), dsec = f"Dev {epoch}") as pbar:
      for dev_batch in dev_dataloader:
        text, mel_targets, lin_targets = (tensor.to(device) for tensor in dev_batch)
        mel_outputs, lin_outputs, _= model(text, mel_targets)

        #L1 loss
        mel_loss = abs(mel_outputs - mel_targets)
        lin_loss = abs(lin_outputs - lin_targets)
        loss = mel_loss + lin_loss

        dev_loss_stack.append(loss.item())
        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(dev_loss_stack):.3f})")

    print(f"[DEV] EP: {epoch} Avg: {np.mean(dev_loss_stack):.4f}")