# Amyloid plaque morphology classifier

Finds amyloid plaques in fluorescence microscopy and sorts each one by shape into the
three morphologies pathologists name — Diffuse, DenseCore, and Compact. It also drops
every plaque onto a 2D morphology map, so I can look at the whole population as a
continuum instead of three hard bins, which is closer to how the plaques actually behave.

## Pipeline

Four stages, image to label.

1. **Segmentation** (`plaque_classifier/segmentation.py`): a double band-pass filter
   rejects slow tissue-level intensity drift and locks onto plaque-sized objects, then a
   watershed splits plaques that touch.
2. **Not-plaque filter** (`plaque_classifier/cnn.py`): a ResNet50 makes the plaque /
   not-plaque call and throws out the vessels, tissue folds, and debris the segmenter also
   grabs. The only stage that needs PyTorch.
3. **Features** (`plaque_classifier/features.py`): 47 engineered morphology features per
   object: intensity statistics, texture, the radial intensity profile, shape, and a few
   compound features I added to hand a linear model the core-versus-halo signal it can't
   build on its own.
4. **Classification** (`plaque_classifier/classifier.py`): an LDA assigns the class and
   gives the 2D projection. I read LD1 as a maturity axis, Diffuse through Compact, and LD2
   as coredness.

The shape call goes to engineered features and a linear model, rather than a second CNN,
because I want to be able to point at *why* a plaque was called Compact. Every one of the
47 features has a physical meaning, and the LDA weights tell me which ones drove each
class. The CNN only handles the coarse plaque/not-plaque split, where the decision is easy
and there's nothing subtle to explain.

Feature selection is deliberate. Each feature gets ranked by ANOVA F-test and by mutual
information, then greedy forward selection runs under image-level cross-validation
(`scripts/evaluate_features.py`). And every split (training, CV, evaluation) is grouped
by source image, so tiles from one slide never land in both folds. Skip that and the
accuracy number is fiction.

## Running the demo

```
pip install -r requirements.txt
```

The core pipeline needs numpy, scipy, scikit-image, scikit-learn, pandas, matplotlib,
Pillow, and tifffile. torch and torchvision are commented out in `requirements.txt` —
they're heavy and only the CNN stage touches them.

Build the synthetic tiles and whole images, train the LDA on the tiles, then predict on
the images:

```
python scripts/make_synthetic_data.py --out data
python scripts/train.py --data data --out models/morphology_lda.pkl
python scripts/predict.py --images data/images --model models/morphology_lda.pkl --out predictions
```

`train.py` prints a cross-validated report on how well the three classes separate.
`predict.py` writes one row per plaque, a per-image summary, an overlay, and a scatter of
the continuum. The synthetic generator (`plaque_classifier/synthetic.py`) is what lets this
run end to end on a clean machine; swap in real tiles and a `labels.csv` and nothing else
changes.

For the real end-to-end number — segmentation, features, and classification scored
together against the generator's ground truth, reported as detection recall plus the
morphology confusion:

```
python scripts/evaluate_pipeline.py --images data/images --model models/morphology_lda.pkl --truth data/images_truth.csv
```

The CNN filter is optional. Once torch is in, train it and switch it on at predict time:

```
python scripts/train_cnn.py --data data --out models/binary_cnn_resnet50.pth
python scripts/predict.py --images data/images --model models/morphology_lda.pkl --out predictions --cnn models/binary_cnn_resnet50.pth
```

## Your own data

Training wants a folder with a `tiles/` subfolder and a `labels.csv`: one row per tile, a
`tile` column with the filename, a `label` column (Diffuse, DenseCore, Compact, or
NotPlaque), and an optional `image` column naming the source slide so the CV grouping
holds. One centered object per tile.

Prediction wants a folder of whole images. TIFF and PNG both work, and a multi-page TIFF is
max-projected. Object size limits and the segmentation scales are in microns, so pass
`--px_um` with your objective's microns-per-pixel; objects between 10 and 200 µm equivalent
diameter are kept. The 47 features and the LDA carry from synthetic to real unchanged — the
one piece worth retraining on real tiles is the binary CNN.

## Layout

```
plaque_classifier/
    features.py       47 morphology features per object
    segmentation.py   candidate detection, tile extraction, overlays
    classifier.py     LDA training, 2D continuum, save/load, prediction
    cnn.py            binary plaque / not-plaque ResNet50 (needs PyTorch)
    synthetic.py      synthetic plaque and image generator
    pipeline.py       whole-image prediction, end to end
scripts/
    make_synthetic_data.py   build the synthetic tiles and images
    train.py                 train the LDA
    predict.py               classify plaques in whole images
    evaluate_pipeline.py     score the whole pipeline against ground truth
    evaluate_features.py     feature ranking and forward selection
    train_cnn.py             train the ResNet50 (needs PyTorch)
```
