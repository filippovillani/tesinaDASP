import torch
import os
import torchaudio
import torchvision
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#################### Directories and file paths ##################################
main_dir = r'D:\Università\MAGISTRALE\DigitalAudioSignalProcessing\Tesina'
test_data_dir = os.path.join(main_dir, 'data')
images_dir = os.path.join(test_data_dir, 'test_A')
audio_dir = os.path.join(test_data_dir, 'test_B')
##################################################################################

#%%
# Audio files must be downsampled to 22.050 kHz and converted to monoaural.
sr = 22050
max_duration = 5.94 # in seconds
max_len = int(sr * max_duration)

for audio in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, audio)
    audio, old_sr = torchaudio.load(audio_path)
    # Rechannel
    if (audio.shape[0] == 2):
        audio = audio.mean(dim=0)
    
    # Resample
    if (old_sr != sr):
        audio = torchaudio.transforms.Resample(old_sr, sr)(audio)
    
    # Pad or trunc
    audio_len = len(audio)
    
    if audio_len > max_len:
        audio = audio[:max_len]
    
    elif audio_len < max_len:
        padding_len = max_len - audio_len
        audio = torch.cat([audio, torch.zeros(padding_len)])
    
    torchaudio.save(audio_path, audio.unsqueeze(dim=0), sr)
    
#%%
for image in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image)
    image = torchvision.io.read_image(image_path)
    image = torchvision.transforms.Resize([224,224])(image)

    torchvision.utils.save_image((image / 255.).float(), image_path)