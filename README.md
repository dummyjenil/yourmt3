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
import yourmt3
from huggingface_hub import hf_hub_download
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = yourmt3.YMT3(hf_hub_download("shethjenil/Audio2Midi_Models","YPTF+MoE+Multi 1.pt"),"YPTF+MoE+Multi","32" if device == "cpu" else "16",torch.device(device))
gr.Interface(lambda  path,batch_size,progress=gr.Progress():model.predict(path,lambda  i,total:progress((i,total)),batch_size),[gr.Audio(type="filepath",label="Audio"),gr.Number(8,label="Batch Size")],gr.File(label="midi")).launch()
```