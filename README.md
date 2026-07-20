# Amyloid Plaque Morphology Classifier

I built this to classify the morphology of amyloid plaques in fluorescence microscopy. Given an image it finds each plaque and sorts it into one of the three shapes pathologists describe, Diffuse, DenseCore, and Compact. It also places every plaque on a 2D morphology continuum so I can look at the whole population instead of only the discrete labels.

The repository ships with a synthetic data generator, so the whole pipeline runs end to end without any real microscopy. I swap in real images and labels when I want to run it for real.

## How it works

The pipeline has four stages.

- Segmentation. I use a double band-pass filter to reject slow tissue-level intensity changes and lock onto plaque-sized objects, then a watershed to split plaques that touch. This lives in `plaque_classifier/segmentation.py`.
- Not-plaque filter. A ResNet50 decides plaque versus not-plaque and drops the vessels, tissue folds, and debris that the segmenter also picks up. This is the only stage that needs PyTorch, and it lives in `plaque_classifier/cnn.py`.
- Feature extraction. I compute 47 engineered morphology features per object. They cover intensity statistics, texture, the radial intensity profile, object shape, and a handful of compound features I added to hand a linear model the core-versus-halo signal it cannot learn on its own. This lives in `plaque_classifier/features.py`.
- Classification. A Linear Discriminant Analysis assigns the class and gives the 2D projection, where I read LD1 as a maturity axis running from Diffuse to Compact and LD2 as a coredness axis. This lives in `plaque_classifier/classifier.py`.

I keep the feature set honest by ranking every feature with an ANOVA F-test and with mutual information, then running greedy forward selection under image-level cross validation. That code is in `scripts/evaluate_features.py`.

## Why this design

I wanted a classifier I can fully explain. A pure CNN would be a black box, so I use the CNN only for the easy plaque versus not-plaque call, where I do not need to explain much, and I hand the subtle morphology decision to engineered features plus a linear model. Every feature has a physical meaning, and the LDA weights show me which features drive each class. The same LDA gives the continuum, which matches how these plaques really behave, where the shapes grade into each other rather than falling into clean bins.

Cross validation is grouped by source image everywhere, so tiles from one image never land in both the train fold and the test fold. That is the part that keeps the reported accuracy honest.

## Repo layout

```
plaque_classifier/
    features.py       engineered morphology features (47 per object)
    segmentation.py   candidate detection, tile extraction, overlay rendering
    classifier.py     LDA training, 2D continuum, save and load, prediction
    cnn.py            binary plaque / not-plaque ResNet50 (needs PyTorch)
    synthetic.py      synthetic plaque and image generator
    pipeline.py       whole-image prediction (segment, filter, features, classify)
scripts/
    make_synthetic_data.py   build the synthetic tiles and images
    train.py                 train the LDA morphology classifier
    predict.py               classify plaques in whole images
    evaluate_pipeline.py     score the whole pipeline against ground truth
    evaluate_features.py     feature ranking and forward selection
    train_cnn.py             train the binary ResNet50 (needs PyTorch)
```

## Install

```
pip install -r requirements.txt
```

The core pipeline needs numpy, scipy, scikit-image, scikit-learn, pandas, matplotlib, Pillow, and tifffile. The not-plaque CNN also needs torch and torchvision, which I left commented out in `requirements.txt` because they are heavy and nothing else in the pipeline depends on them.

## Run the synthetic demo

First I generate the synthetic tiles and whole images.

```
python scripts/make_synthetic_data.py --out data
```

Then I train the morphology classifier on the tiles. The run prints a cross-validated report so I can see how well the three classes separate.

```
python scripts/train.py --data data --out models/morphology_lda.pkl
```

Then I predict on the whole synthetic images. This writes one row per plaque, a per-image summary, an overlay for each image, and a scatter of the morphology continuum.

```
python scripts/predict.py --images data/images --model models/morphology_lda.pkl --out predictions
```

The tile training run scores the classifier on its own. To score the whole pipeline end to end, segmentation and features and classification together, I match predictions against the ground truth that the generator wrote and report detection recall plus the morphology confusion. This is the honest end-to-end number.

```
python scripts/evaluate_pipeline.py --images data/images --model models/morphology_lda.pkl --truth data/images_truth.csv
```

To reproduce the feature ranking and the forward selection I run this.

```
python scripts/evaluate_features.py --data data
```

The CNN stage is optional and needs PyTorch. Once torch is installed I train the filter and then switch it on during prediction.

```
python scripts/train_cnn.py --data data --out models/binary_cnn_resnet50.pth
python scripts/predict.py --images data/images --model models/morphology_lda.pkl --out predictions --cnn models/binary_cnn_resnet50.pth
```

## Using my own data

For training I point `--data` at a folder that holds a `tiles` subfolder and a `labels.csv`. The CSV has one row per tile, a column named `tile` with the image filename, a column named `label` with one of Diffuse, DenseCore, Compact, or NotPlaque, and an optional column named `image` naming the source slide so the cross-validation splits stay grouped. Each tile is a single centered object.

For prediction I point `--images` at a folder of whole images. TIFF and PNG both work, and a multi-page TIFF is max-projected. Object size limits and the segmentation scales are set in microns, so I pass `--px_um` with the microns per pixel of my objective. Objects between 10 and 200 microns equivalent diameter are kept.

## Running it on real data

The synthetic set is only there so the pipeline runs out of the box. On real data I point the loader at my own images and labels, set `--px_um` to my microscope calibration, and retrain the binary CNN on real tiles. The 47 morphology features and the LDA carry over unchanged.
