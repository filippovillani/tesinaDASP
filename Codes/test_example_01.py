import torch
import os
from model import Image2Reverb
from dataset import Image2ReverbDataset
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from utilities import plot_frequency_response, convolve_speech_and_ir, plot_spectrograms

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#################### Directories and file paths ##################################
main_dir = r'D:\Università\MAGISTRALE\DigitalAudioSignalProcessing\Tesina'
pretrained_model = os.path.join(main_dir, 'model.ckpt')
test_data_dir = os.path.join(main_dir, 'data')
ir_true_dir = os.path.join(test_data_dir, 'test_B')
encoder_path = os.path.join(main_dir, 'resnet50_places365.pth.tar')
depthmodel_path = os.path.join(main_dir, 'mono_odom_640x192')
speech_dir = os.path.join(main_dir, 'Voice_examples')
speech_path = os.path.join(speech_dir, 'voice01.wav')

output_dir = os.path.join(main_dir, 'image2reverb_outputs')
##################################################################################
# Building the dataset
test_set = Image2ReverbDataset(test_data_dir, "test", 'stft')
test_dataset = torch.utils.data.DataLoader(test_set, batch_size=1)

ir_predicted = []

def save_outputs(examples, spectrograms, audio, input_images, input_depthmaps):
    for i, example in enumerate(examples):
        print("\n Processing example %d: %s." % (i, example))
        d = os.path.join(output_dir, example)
        if not os.path.isdir(d):
            os.makedirs(d)
        sf.write(os.path.join(d, "%s_IR_pred.wav" % example), audio[i], 22050)
        plt.imsave(os.path.join(d, "input.png"), input_images[i])
        plt.imsave(os.path.join(d, "depth.png"), input_depthmaps[i])
        ir_predicted.append(audio[i])

# Initialize the model and load the pretrained weights
model = Image2Reverb(encoder_path, depthmodel_path, test_callback=save_outputs).to(device)
m = torch.load(pretrained_model, device)
model.load_state_dict(m['state_dict'])

# Test
trainer = Trainer(gpus=1)
trainer.test(model, test_dataset)

# Plot spectrograms, frequency responses and save convolved speeches
ir_true_path = []

for i, out_path in enumerate(os.listdir(output_dir)):
    ir_true_path = os.path.join(ir_true_dir, os.listdir(ir_true_dir)[i])
    ir_true, _ = librosa.load(ir_true_path, sr=22050)
    plot_frequency_response(ir_true, ir_predicted[i], os.path.join(output_dir, out_path))
    plot_spectrograms(ir_true, ir_predicted[i], os.path.join(output_dir, out_path))
    speech, _ = librosa.load(speech_path, sr=22050)
    convolve_speech_and_ir(ir_true, ir_predicted[i], speech, os.path.join(output_dir, out_path))