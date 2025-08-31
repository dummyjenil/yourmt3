## *Model Types*

* YMT3+

* YPTF+Single

* YPTF+Multi

* YPTF+MoE+Multi 1

* YPTF+MoE+Multi 2
---
---
```bash
pip  install  yourmt3
```

```python
import gradio as gr
from yourmt3 import YMT3
from huggingface_hub import hf_hub_download
model = YMT3(hf_hub_download("shethjenil/Audio2Midi_Models","YMT3+.pt"),"YMT3+")
gr.Interface(lambda path,batch_size,progress=gr.Progress():model.predict(path,batch_size,lambda i,total:progress((i,total)),),[gr.Audio(type="filepath",label="Audio"),gr.Number(8,label="Batch Size")],gr.File(label="midi")).launch()
```

# Based On
[Research paper](https://arxiv.org/abs/2407.04822)

[Colab Code](https://colab.research.google.com/drive/1AgOVEBfZknDkjmSRA7leoa81a2vrnhBG?usp=sharing)