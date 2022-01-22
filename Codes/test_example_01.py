import torch
import os
from model import Image2Reverb
from dataset import Image2ReverbDataset
from pytorch_lightning import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#################### Directories and file paths ##################################
main_dir = r'D:\Università\MAGISTRALE\DigitalAudioSignalProcessing\Tesina'
pretrained_model = os.path.join(main_dir, 'model.ckpt')
test_data_dir = os.path.join(main_dir, 'data')
encoder_path = os.path.join(main_dir, 'resnet50_places365.pth.tar')
depthmodel_path = os.path.join(main_dir, 'mono_odom_640x192')
##################################################################################
# Building the dataset
test_set = Image2ReverbDataset(test_data_dir, "test", 'stft')

# Initialize the model and load the pretrained weights
model = Image2Reverb(encoder_path, depthmodel_path).to(device)
m = torch.load(pretrained_model, device)
model.load_state_dict(m['state_dict'])

# Test
trainer = Trainer(gpus=1)
trainer.test(model, test_set)
