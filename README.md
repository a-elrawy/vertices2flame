## **Vertices to Flame**
This repo contains a 3d vertices to flame parameters (pose and expression) model. This model convert vertices to blend shapes, flame expression blend shapes and jaw rotation parameters. It works as a flame inverter to extract back the parameters from the vertices


## **Environment**
- Linux
- Python 3.6+
- Pytorch 2.0.1

Other necessary packages:
```
pip install -r requirements.txt
```
- unzip (for dataset extraction)

## **Usage**
```python
import torch
from vertices2flame import FlameInverter
inverter = FlameInverter(from_pretrained=True)
vert = torch.zeros((1, 1, 15069))
pose, exp = inverter(vert)
```

## **Dataset Preparation**
Download "FLAME_sample.ply" from [voca](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `dataset/model`.

#### VoxCeleb 
Request the Flame model from [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/). Place the download files `flame_static_embedding.pkl`, `flame_dynamic_embedding.npy` and `generic_model.pkl` in `dataset/model`
Download the dataset from the bucket:
```
 python download_dataset.py audio2face /path/to/credential.json
```

## **Training**

The training operation shares a similar command:
```
sh train.sh FLAME_1 config/train.yaml vox 
```


## **Acknowledgement**
We heavily borrow the code from
[Codetalker]('https://github.com/Doubiiu/CodeTalker'),
