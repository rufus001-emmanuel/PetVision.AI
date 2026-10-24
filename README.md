# PetVision AI — Cat, Dog, or Neither
## Project overview
This project classifies an image into one of three classes: cat, dog, or neither.
## Approach
- Transfer learning with MobileNetV2
- Image resizing and normalization
- Train / validation / test split
- Evaluation using accuracy, precision, recall, F1-score, and a confusion matrix
## Dataset
Oxford-IIIT Pet and Caltech-101. See their official dataset pages.
The "neither" class uses selected Caltech-101 categories other than cats and dogs.
## Results
- Test accuracy: [enter your actual result]
- Main observations: [explain what worked and what failed]
- Confusion matrix: [include image or link]
## Limitations
The "neither" class does not represent every object in the real world.
Dataset differences and unfamiliar images can cause mistakes.
Model scores are not guarantees of correctness.
## How to run
Open PetVision_AI.ipynb in Google Colab and run the cells in order.
Optional app: install requirements and run `streamlit run app.py`.
