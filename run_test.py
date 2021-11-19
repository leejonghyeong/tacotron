from tqdm.notebook import tqdm
import numpy as np
from matplotlib import pyplot as plt
import librosa


def run_test(model,
             device,
             test_dataloader):
  '''
  choose one test example and show attention alignment
  '''

  #run_test: compute loss and show attention alignment
  model.eval()
  test_loss_stack = []
  with tqdm(total = len(test_dataloader)) as pbar:
    for test_batch in test_dataloader:
      text, mel_targets, lin_targets = (tensor.to(device) for tensor in test_batch)
      mel_outputs, lin_outputs, _ = model(text, mel_targets)

      #L1 loss
      mel_loss = abs(mel_outputs - mel_targets)
      lin_loss = abs(lin_outputs - lin_targets)
      loss = mel_loss + lin_loss

      test_loss_stack.append(loss.item())
      pbar.update(1)
      pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(test_loss_stack):.3f})")

    print(f"[DEV] Avg: {np.mean(test_loss_stack):.4f}")

def get_voice(model,
              text,
              mel):
  _, _, voice = model(text, mel)
  return voice

def show_attn_align(model,
                    text,
                    mel):
  #get attention weight
  attn = model.get_attn(text, mel)

  #show plot
  fig = plt.figure(figsize=(8,6))
  plt.pcolormesh(attn)
  plt.title("Attention Alignment")
  plt.xlabel("Decoder timesteps")
  plt.ylabel("Encoder states")
  plt.colorbar()
  plt.show()

def compare_wav_plot(model,
                     text,
                     mel,
                     original,
                     sr):
  _,_, voice = model(text, mel)

  fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
  librosa.display.waveshow(original, sr=sr, color='b', ax=ax[0])
  ax[0].set(title='Original', xlabel=None)
  ax[0].label_outer()
  librosa.display.waveshow(voice, sr=sr, color='g', ax=ax[1])
  ax[1].set(title='Tacotron reconstruction', xlabel=None)
  ax[1].label_outer()

  librosa.output.write_wav('\\sample\\sample_original.wav', original, sr)
  librosa.output.write_wav('\\sample\\sample_tacotron.wav', voice, sr)