# RVC Inference


Translations provided by GPT-4.

This project is a lightweight, fast, and memory efficient api that runs v1/v2 RVC models. It is intended for use in production environments and compatibility with existing codebases.
It makes integrating RVC as a stage in a pipeline or workflow easy. Installation is quick using pip and should be compatible with
Linux/Windows/Mac and the latest python versions.
## Install
If using Python 3.11+ install the fairseq fork first as fairseq is not yet compatible with 3.11. (Will take a minute).


Pip install the repo like below and all dependencies will be installed automatically.
```bash
pip uninstall inferrvc
pip install inferrvc --no-cache-dir
```
By default pypi installs the pytorch cpu build. To install for gpu using Nvidia or AMD, visit https://pytorch.org/get-started/locally/ and pip install `torch` and `torchaudio` with gpu _before_ installing this library.

Support should be available for Python 3.8-3.12 but only 3.11 was tested. If there are any problems with installation or compatibility please open an issue and I'll push out fixes.
PR's with fixes and improvements are welcome.

## Usage
First set the optional environment variables:
```python
import os
os.environ['RVC_MODELDIR']='path/to/rvc_model_dir' #where model.pth files are stored.
os.environ['RVC_INDEXDIR']='path/to/rvc_index_dir' #where model.index files are stored.
#the audio output frequency, default is 44100.
os.environ['RVC_OUTPUTFREQ']='44100'
#If the output audio tensor should block until fully loaded, this can be ignored. But if you want to run in a larger torch pipeline, setting to False will improve performance a little.
os.environ['RVC_RETURNBLOCKING']='True'
```
**Notes on environment variables:**
- Both `RVC_OUTPUTFREQ` and `RVC_RETURNBLOCKING` set defaults for the `RVC` class, but they can be overriden per instance with `self.outputfreq` and `self.returnblocking`.  
- Setting `RVC_OUTPUTFREQ` to `None` will disable standard resampling and return the model's native sample rate.  
- If you do not set `RVC_INDEXDIR` the `RVC` class will fallback to `RVC_MODELDIR` and lastly the absolute path of the model directory `os.path.dirname(model_path)`.  
- If you do not set `RVC_MODELDIR` then arg `model` must be an absolute path.

Load models:
```python
from inferrvc import RVC
whis,obama=RVC('Whis.pth',index='added_IVF1972_Flat_nprobe_1_Whis_v2'),RVC(model='obama')

print(whis.name)
print('Paths',whis.model_path,whis.index_path)
print(obama.name)
print('Paths',obama.model_path,obama.index_path)
```
```text
Model: Whis, Index: added_IVF1972_Flat_nprobe_1_Whis_v2
Paths Z:\Models\RVC\Models\Whis.pth Z:\Models\RVC\Indexes\added_IVF1972_Flat_nprobe_1_Whis_v2.index
Model: obama, Index: obama
Paths Z:\Models\RVC\Models\obama.pth Z:\Models\RVC\Indexes\obama.index
```

Run Inferencing:
```python
from inferrvc import load_torchaudio
aud,sr = load_torchaudio('path/to/audio.wav')

paudio1=whis(aud,f0_up_key=6,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.75)
paudio2=obama(aud,5,output_device='cpu',output_volume=RVC.MATCH_ORIGINAL,index_rate=.9)

import soundfile as sf

sf.write('path/to/audio_whis.wav',paudio1,44100)
sf.write('path/to/audio_obama.wav',paudio2,44100)
```
[Whis example.](./docs/audio_whis.wav)  
[Obama example.](./docs/audio_obama.wav)

### Changes from the original repo:
 - Removed most code not related to inferencing. Now much fewer dependencies.
 - Made a streamlined inference class and pipeline.
 - Performance and memory efficiency improvements.
 - Generic models are now managed by `huggingface_hub` and cached through the path `HF_HOME` environment variable.
 - Flexible referencing of RVC model directory and files.
 - Disabled the butterworth filter by default as there is usually no difference and might slightly reduce quality. Can be enabled with `inferrvc.pipeline.enable_butterfilter=True`.

