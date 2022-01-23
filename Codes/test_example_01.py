import torch
import os
from model import Image2Reverb
from dataset import Image2ReverbDataset
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#################### Directories and file paths ##################################
main_dir = r'D:\Università\MAGISTRALE\DigitalAudioSignalProcessing\Tesina'
pretrained_model = os.path.join(main_dir, 'model.ckpt')
test_data_dir = os.path.join(main_dir, 'data')
encoder_path = os.path.join(main_dir, 'resnet50_places365.pth.tar')
depthmodel_path = os.path.join(main_dir, 'mono_odom_640x192')

test_dir = os.path.join(main_dir, 'image2reverb_outputs')
##################################################################################
# Building the dataset
test_set = Image2ReverbDataset(test_data_dir, "test", 'stft')
test_dataset = torch.utils.data.DataLoader(test_set, batch_size=1)

def save_outputs(examples, spectrograms, audio, input_images, input_depthmaps):
    for i, example in enumerate(examples):
        print("\n Processing example %d: %s." % (i, example))
        d = os.path.join(test_dir, example)
        if not os.path.isdir(d):
            os.makedirs(d)
        plt.imsave(os.path.join(d, "spec.png"), spectrograms[i])
        sf.write(os.path.join(d, "%s.wav" % example), audio[i], 22050)
        plt.imsave(os.path.join(d, "input.png"), input_images[i])
        plt.imsave(os.path.join(d, "depth.png"), input_depthmaps[i])

# Initialize the model and load the pretrained weights
model = Image2Reverb(encoder_path, depthmodel_path, test_callback=save_outputs).to(device)
m = torch.load(pretrained_model, device)
model.load_state_dict(m['state_dict'])

# Test
trainer = Trainer(gpus=1)
trainer.test(model, test_dataset)
