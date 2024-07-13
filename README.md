# Panoptic-Segmentation-on-Dendrites-and-Dendritic-Spines
This repository extends the <a href="https://github.com/ankilab/DeepD3">DeepD3</a> project by adding panoptic segmentation capabilities for the detailed analysis and quantification of dendrites and dendritic spines. It utilizes Mask R-CNN for instance segmentation of dendritic spines and FCN-ResNet for semantic segmentation of dendrites.

# Steps to Get Started:
### Step 1: Clone this repository
```
git clone https://github.com/sahil-sharma-50/Panoptic-Segmentation-on-Dendrites-and-Dendritic-Spines.git
```
### Step 2: Download instance and semantic model:
<ol>
  <li>Instance Segmentation Model for Spines: <a href='https://faubox.rrze.uni-erlangen.de/getlink/fiEfTXy8DJhqCzCksmgiC6/spines_model.pt'>MaskRCNN FAUBox</a></li>
  <li>Semantic Segmentation Model for Dendrites: <a href='https://faubox.rrze.uni-erlangen.de/getlink/fi7iUL8cVWUsA5w9ZFLj2A/dendrite_model.pt'>FCN_ResNet50</a></li>
</ol>

`For example: Save these models in .\src\panoptic_inference`

### Step 3: Install the `requirements.txt` file:
```
pip install -r requirements.txt
```
### Step 4: Run panoptic_inference.py
```
cd .\src\panoptic_inference\
python panoptic_inference.py
```
