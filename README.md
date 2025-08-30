```bash
pip  install  yourmt3
```
---
## *Model Types*

* YMT3+

* YPTF+Single

* YPTF+Multi

* YPTF+MoE+Multi 1

* YPTF+MoE+Multi 2

* YMT3+MusicFM
---

```python
import gradio as gr
from yourmt3 import YMT3
from huggingface_hub import hf_hub_download
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YMT3(hf_hub_download("shethjenil/Audio2Midi_Models","YPTF+MoE+Multi 2.pt"),"YPTF+MoE+Multi","32" if device == "cpu" else "16",torch.device(device))
gr.Interface(lambda path,batch_size,confidence_threshold,instrument,progress=gr.Progress():model.predict(path,batch_size,confidence_threshold,instrument,lambda i,total:progress((i,total)),),[gr.Audio(type="filepath",label="Audio"),gr.Number(8,label="Batch Size"),gr.Slider(0,1,0.7,step=0.01,label="Confidence Threshold"),gr.Dropdown(["default","singing-only","drum-only"])],gr.File(label="midi")).launch()
```