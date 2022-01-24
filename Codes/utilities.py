import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from scipy.fft import fft
from scipy.signal import convolve

def plot_frequency_response(true_ir, predicted_ir, output_dir):
    true_fr = np.abs(fft(true_ir))
    true_fr = 10*np.log10(true_fr)

    predicted_fr = np.abs(fft(predicted_ir))
    predicted_fr = 10*np.log10(predicted_fr)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.semilogx(true_fr[:int(0.5*len(true_fr))])
    plt.xlim([10, 5e4])
    plt.grid(which='both')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Recorded Frequency Response')

    plt.subplot(2,1,2)
    plt.semilogx(predicted_fr[:int(0.5*len(predicted_fr))])
    plt.xlim([10, 5e4])
    plt.grid(which='both')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.title('Predicted Frequency Response')

    plt.subplots_adjust(left=0.12,
                        bottom=0.14, 
                        right=0.96, 
                        top=0.94, 
                        wspace=0.2, 
                        hspace=0.42)
    
    plt.savefig(os.path.join(output_dir, "frequency_response.png"))
    
def convolve_speech_and_ir(ir_true, ir_predicted, speech, output_dir, sr=22050):
    
    convolved_voice_true = convolve(speech, ir_true)
    convolved_voice_true /= ((max(convolved_voice_true) - min(convolved_voice_true)) / 2)
    sf.write(os.path.join(output_dir, 'wet_voice_true.wav'), convolved_voice_true, sr)
    
    convolved_voice_pred = convolve(speech, ir_predicted)
    convolved_voice_pred /= ((max(convolved_voice_pred) - min(convolved_voice_pred)) / 2)
    sf.write(os.path.join(output_dir, 'wet_voice_predicted.wav'), convolved_voice_pred, sr)  
    