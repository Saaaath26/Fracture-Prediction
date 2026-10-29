# Scientific validity review

The intended study is to learn fracture-associated radiographic shape relationships and, only later, to ask whether any verified relationship could inform an external limb photograph. The code now keeps those stages apart.

## What FracAtlas can support

FracAtlas can supervise fracture presence, fracture boxes, and fracture instance masks. COCO is the conversion source. The official fractured split is 574 / 82 / 63, with 3,364 non-fractured images unassigned. That split cannot measure specificity on normal cases.

A derived split of 559 / 68 / 63 removes images that are unsafe to learn from automatically:

- 59 truncated non-fracture JPEGs
- 16 images in eight train/validation pairs that are the same decoded radiograph with different boxes
- 13 images in six within-split groups with the same property

This derived split is a leakage control for known exact-pixel copies. It is not a patient-level split. Metadata has no patient id. Repeated views can still cross the partition. One hundred seventeen dHash collisions are review candidates only.

## What the contour code is

Classical extraction builds a bright Otsu candidate or a low-intensity foreground envelope. It keeps external contours, rejects border-truncated and near-full-frame components, and records abstention. It does not run Canny, and it does not separate overlapping bones. Quality flags say identity is unverified and confidence is uncalibrated. There is no numeric score that could be quoted as a probability.

Geometry is in the annotation frame. Areas and lengths are pixels unless a caller supplies verified pixel spacing. `circularity`, `solidity`, `aspect_ratio`, and `turning_angle_std_rad` are scale-free. Centroids can be divided by the frame size. `turning_angle_std_rad` is the standard deviation of turning angles on a closed polygon. It is zero for a regular polygon. It is not a clinical curvature in millimeters. Orientation is the major axis of the minimum-area rectangle, modulo 180 degrees. A square does not have a stable clinical axis.

Every exported record states that the measurement is a radiographic projection. Position, view, and scale can explain an apparent shape difference. Dataset columns for hand, leg, hip, shoulder, view, hardware, and multiscan can be copied into the analysis table when a reviewer puts them in the manifest. They are controls, not fracture labels.

The signed distance from the fracture centroid to a candidate boundary, and the fraction of fracture pixels inside that candidate, use the ground-truth fracture mask. They are descriptive. The presence classifier refuses them as inputs.

## What is blocked

Verified bone or soft-tissue masks are not in this dataset. U-Net and nnU-Net preparation require `verified=true`, an annotation source, a binary mask aligned to the image, and a single anatomical target. Fracture polygons do not satisfy that. No masks were synthesized.

The Siamese network is an experimental pair classifier. It needs an independently defined pair label and both patient ids. No such labels were invented.

Smartphone or video limb shape is a different representation. Radiograph measurements do not establish an external silhouette. The schema records that transfer as experimental and unimplemented.

## Coordinate decision for one image

`IMG0002628.jpg` is the only file whose stored size disagrees with COCO. EXIF orientation 8 produces 454×373, which matches COCO and VOC. The derived PNG uses that oriented frame. The original JPEG was not rewritten. Images whose stored size already matches the annotation keep their stored pixels even if an orientation tag is present. Applying every tag would rotate labels off the annotated anatomy.

## Questions a later experiment could answer

1. Are candidate contour measurements associated with fracture presence after the image is restricted to one anatomical site and one view?
2. Do those associations differ by the hand, leg, hip, or shoulder flags already in `dataset.csv`?
3. Do they remain after controlling for centroid position and image scale?
4. Do they improve on a simple baseline that uses only those confounders?
5. Does a patient-disjoint, externally reviewed cohort reproduce the result?

None of those questions were fit or tested here. No performance number in the repository is a model result.
