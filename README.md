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
---

```python
import gradio as gr
from yourmt3 import YMT3
from huggingface_hub import hf_hub_download
model = YMT3(hf_hub_download("shethjenil/Audio2Midi_Models","YMT3+.pt"),"YMT3+")
gr.Interface(lambda path,batch_size,confidence_threshold,instrument,progress=gr.Progress():model.predict(path,batch_size,confidence_threshold,instrument,lambda i,total:progress((i,total)),),[gr.Audio(type="filepath",label="Audio"),gr.Number(8,label="Batch Size"),gr.Slider(0,1,0.7,step=0.01,label="Confidence Threshold"),gr.Dropdown(["default","singing-only"])],gr.File(label="midi")).launch()
```