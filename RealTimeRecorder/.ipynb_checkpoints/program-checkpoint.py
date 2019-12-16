import os
import sys
import pyaudio
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import librosa

from models import MMDenseNet
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PyInstaller.utils.hooks import get_package_paths

# get path of soundfile
sfp = get_package_paths('soundfile')

# add the binaries
bins = os.path.join(sfp[0], "_soundfile_data")
datas = [(bins, "_soundfile_data")]
binaries = []

## import model
model = MMDenseNet(drop_rate=0.25,bn_size=4,k1=10,l1=3,k2=14,l2=4,attention = 'CBAM')
model.cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("weight.pth"))

CHUNK = 1024
ANALYSIS_FRAME = 64
ANALYSIS_SAMPLE = 64*128
RATE = 16000

n_fft = 512
hop_length = 128
DIFF_MEL = CHUNK//hop_length

DEFAULT_SAMPLE = np.array([0]*(ANALYSIS_SAMPLE-CHUNK))

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        
        self.recorder = pyaudio.PyAudio()
        self.is_recording = False
    def setupUI(self):
        record_but = QPushButton("녹음하기", self)
        record_but.clicked.connect(self.record)
        
        stop_but = QPushButton("녹음그만두기", self)
        stop_but.clicked.connect(self.stop)
    
        layout = QHBoxLayout()
        layout.addWidget(record_but)
        layout.addWidget(stop_but)
        self.setLayout(layout)
    
    def plot(self):
        ## plot self.accumulated_egg_mel
        return 
    
    def record(self):
        self.accumulated_audio = None
        self.accumulated_speech_mel = None
        self.accumulated_egg_mel = None
        self.is_recording = True
        self.stream=self.recorder.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                frames_per_buffer=CHUNK,input_device_index=2)
        
        while(self.is_recording):
            data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
            data = data.astype('float32')/32768.

            if self.accumulated_audio is None:
                self.accumulated_audio = np.concatenate([DEFULAT_SAMPLE,data])
            else:
                self.accumulated_audio = np.concatenate([self.accumulated_audio,data])

            if self.accumulated_speech_mel is None:
                mel = stft_to_mel(librosa.core.stft(self.accumulated_audio,n_fft=n_fft, hop_length=hop_length,center=False))
                self.accumulated_speech_mel = mel
            else:
                mel = stft_to_mel(librosa.core.stft(self.accumulated_audio[-(CHUNK+n_fft-hop_length):],n_fft=n_fft, hop_length=hop_length,center=False))
                self.accumulated_speech_mel = np.concatenate([self.accumulated_speech_mel,mel])

            model_input = self.accumulated_speech_mel[np.newaxis,:,-ANALYSIS_FRAME:]
            model_input = torch.Tensor(model_input).cuda()

            model_output = model(model_input).cpu().detach().numpy()[0]

            if self.accumulated_egg_mel is None:
                self.accumulated_egg_mel = model_output[:,-DIFF_MEL:]
            else:
                self.accumulated_egg_mel = np.concatenate([self.accumulated_egg_mel,model_output[:,-DIFF_MEL:]])
            
            self.plot()
            
    def stop(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.recorder.terminate()

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return np.log(np.maximum(x,clip_val) * C)

def stft_to_mel(stft):
    yS = np.abs(stft)
    yS = librosa.feature.melspectrogram(S=yS,sr=16000,n_mels=80,n_fft=512, hop_length=128,fmax=8192,fmin=60)
    yS = -dynamic_range_compression(yS)
    return yS

if __name__=="__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
        
#         print(int(np.average(np.abs(data))))