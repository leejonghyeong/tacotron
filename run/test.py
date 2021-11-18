from ..data.utils import TacotronDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np
import random

def run_test(n_epochs,
              batch_size,
              model,
              device,
              text_test, 
              lin_test,
              mel_test):
  '''
  choose one test example and show attention alignment
  '''
  #load dataset
  test_dataset = TacotronDataset(text_test, lin_test, mel_test)
  test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                collate_fn= collate_fn)

  #run_test: compute loss and update gradient
  for epoch in range(n_epochs):
    model.eval()
    test_loss_stack = []
    with tqdm(total = len(test_dataloader), desc = f"Train {epoch}") as pbar:
      for id, test_batch in enumerate(test_dataloader):
        text, mel_targets, lin_targets = (tensor.to(device) for tensor in test_batch)
        mel_outputs, lin_outputs, _ = model(text, mel_targets)

        #L1 loss
        mel_loss = abs(mel_outputs - mel_targets)
        lin_loss = abs(lin_outputs - lin_targets)
        loss = mel_loss + lin_loss

        test_loss_stack.append(loss.item())
        pbar.update(1)
        pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(test_loss_stack):.3f})")

    print(f"[DEV] EP: {epoch} Avg: {np.mean(test_loss_stack):.4f}")