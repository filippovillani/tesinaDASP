import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from scipy.fft import fft
from scipy.signal import convolve

sr = 44100

main_dir =  r'D:\Università\MAGISTRALE\DigitalAudioSignalProcessing\Tesina'

ir_dir = os.path.join(main_dir, 'data/test_B')
output_dir = os.path.join(main_dir, 'Convolution_outputs')
voices_dir = os.path.join(main_dir, 'Voice_examples')

example_ir_path = os.path.join(ir_dir, 'schubert_small_dc.wav')
example_voice_path = os.path.join(voices_dir, 'voice01.wav')

example_ir, _ = librosa.load(example_ir_path, sr=sr)
example_voice, _ = librosa.load(example_voice_path, sr=sr)

#%%
room_fr = np.abs(fft(example_ir))
room_fr = 10*np.log10(room_fr)

plt.figure()
plt.subplot(2,1,1)
librosa.display.waveshow(example_ir, sr=44100)
plt.title('Impulse Response')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2,1,2)
plt.semilogx(room_fr[:int(0.5*len(room_fr))])
plt.xlim([10, 5e4])
plt.grid(which='both')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.title('Frequency Response')

plt.subplots_adjust(left=0.12,
                    bottom=0.14, 
                    right=0.96, 
                    top=0.94, 
                    wspace=0.2, 
                    hspace=0.42)

#%%
convolved_voice = convolve(example_voice, example_ir)
convolved_voice /= ((max(convolved_voice) - min(convolved_voice)) / 2)

output_path = os.path.join(output_dir, 'schubert_small_voice.wav')
sf.write(output_path, convolved_voice, sr)
print(max(convolved_voice))
#%%
plt.figure()
plt.subplot(2,1,1)
librosa.display.waveshow(example_voice[:sr*3], sr=sr)
plt.title('Dry Voice')
plt.grid()

plt.subplot(2,1,2)
librosa.display.waveshow(convolved_voice[:sr*3], sr=sr)
plt.title('Wet Voice')
plt.grid()

plt.subplots_adjust(left=0.08,
                    bottom=0.14, 
                    right=0.96, 
                    top=0.94, 
                    wspace=0.2, 
                    hspace=0.42)