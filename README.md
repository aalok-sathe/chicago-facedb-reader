# cfd-reader

**A `python3` module to index, pre-process, and supply facial data from the Chicago Face Database (CFD).**

A convenient way to load data from the Chicago Face Database, wrapped in `numpy` arrays, to feed to your neural network.

## installation
From a terminal, run the command:

`[sudo -H] pip3 install cfd-reader`

## usage

In your Python (3.5+) script or interpreter session, run (suggested syntax):

`import cfd_reader as cfdr`

### load face data

`cfdr.load_data(`*[options]*`)`

**where *options* include**:

- `grayscale=[True, False]`:
    - `True`: provides 3 channel BGR output
    - `False`: provides single channel grayscale output
- `train_proportion=x`:
    - `x`: real-valued variable between 0 and 1. `x` proportion of all data will be supplied as training data, and 1-x data will be supplied as test data
- `resize=[False, (x,y)]`:
    - `False`: keeps images in their original resolution
    - `(x,y)`: tuple of integers to resize the image to

**returns**:

`numpy` array of shape `(n, x, y, c)` where `n` are the total number of images in that particular set (train/test), `(x,y)` is the 2D image shape, and `c` is the number of channels. In other words, `(x,y,c)` is the image shape.