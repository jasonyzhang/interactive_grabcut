# Interactive GrabCut

An interactive interface for using OpenCV's GrabCut algorithm for image segmentation.

## Setup

Install dependencies:
```
pip install numpy opencv-python
```

## Usage

```
python interactive_grabcut.py images/car.jpg masks/car.png
```

## Controls

First, select a bounding box around the object being segmented:
```
q: quit/done.
r: reset.
left-click-down: start bounding box.
left-click-up: end bounding box.
```

Mark foreground and background regions on the image. Press `g` to re-run GrabCut:
```
q: quit/done.
r: reset.
f: Switch to marking foreground.
b: Switch to marking background.
u: undo.
g: Re-run GrabCut segmentation.
left-click-down: start marking region.
left-click-up: end marking region.
```

## Relevant Citation

```
@article{rother2004grabcut,
  title={" GrabCut" interactive foreground extraction using iterated graph cuts},
  author={Rother, Carsten and Kolmogorov, Vladimir and Blake, Andrew},
  journal={ACM transactions on graphics (TOG)},
  volume={23},
  number={3},
  pages={309--314},
  year={2004},
  publisher={ACM New York, NY, USA}
}
```