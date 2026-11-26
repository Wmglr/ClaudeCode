#!/usr/bin/env bash
# Reproduit le montage depuis 00_originals (exécuter depuis Portrait_Project/).
# Dépendances : pip install opencv-python-headless scipy pymatting rembg scikit-image psd-tools onnxruntime
# Modèles ONNX (isnet-general-use.onnx) à placer dans /tmp/claude-0/models/ (voir 01_segment.py)
set -e
python3 04_scripts/01_segment.py 00_originals/jeune_homme.jpg 01_masks/jeune isnet-general-use.onnx
python3 04_scripts/01_segment.py 00_originals/duo_reference.jpg 01_masks/duo isnet-general-use.onnx
python3 04_scripts/01c_refine_duo.py        # alpha du duo raffiné (filtre guidé)
python3 04_scripts/02_split_duo.py          # séparation femme / homme (watershed)
python3 04_scripts/01b_matte_young.py       # détourage jeune homme (cheveux par différence de luminance)
python3 04_scripts/03_layers.py             # calques femme / homme décontaminés
python3 04_scripts/05_plate.py              # fond bois prolongé (aplatissement des planches + miroir)
python3 04_scripts/04_compose.py            # placement (D, échelle : geometry.json)
python3 04_scripts/07_grade_render.py       # étalonnage (grade_params.json) + rendu
python3 04_scripts/08_export.py             # PNG / JPG / PSD / calques
