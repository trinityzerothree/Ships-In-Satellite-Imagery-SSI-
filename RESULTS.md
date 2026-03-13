# Experiment Results

All runs: 10 epochs, AdamW (lr=1e-3), batch size 32, 70/15/15 train/val/test split, CPU.

## Summary

| Experiment | Train Size | Test Acc | Test F1 | Test Prec | Test Rec |
|---|---|---|---|---|---|
| No augmentation | 2,800 | 0.9817 | 0.9627 | 0.9793 | 0.9467 |
| On-the-fly augmentation | 2,800 | 0.9483 | 0.9022 | 0.8563 | 0.9533 |
| Offline augmentation (rotations + flips) | 16,800 | 0.9833 | 0.9669 | 0.9605 | 0.9733 |

## Key Takeaways

- **No augmentation** achieved strong results on its own (98.2% accuracy, 0.963 F1), likely because the dataset is relatively small and uniform (80x80 satellite patches).
- **On-the-fly augmentation** (random rotation, flips, color jitter, grayscale, random crop) actually hurt performance. The model trained slower and ended with lower test accuracy (94.8%) and F1 (0.902). The random transforms may be too aggressive for this dataset, distorting features the model needs to classify ships.
- **Offline augmentation** (deterministic 90/180/270 rotations + horizontal/vertical flips) performed best overall, with the highest test accuracy (98.3%) and recall (0.973). Expanding the training set 6x with geometric copies gave the model more data without distorting image quality.

## Confusion Matrices (Test Set)

**No augmentation:**
```
[[447   3]
 [  8 142]]
```

**On-the-fly augmentation:**
```
[[426  24]
 [  7 143]]
```

**Offline augmentation:**
```
[[444   6]
 [  4 146]]
```

Offline augmentation had the fewest total misclassifications (10) and the best balance between false positives (6) and false negatives (4).
