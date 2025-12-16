<p align="center">
  <strong style="color:#d73a49; font-size:1.2em;">
    🚧 WORK IN PROGRESS — Code cleaning and upload in progress 🚧
  </strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-work_in_progress-red" />
</p>


## Machine Learning Model Integrating Computational Pathology to Predict Early Recurrence of Hepatocellular Carcinoma after Resection.

This repository contains the code and supplementary materials for our article:
*Machine Learning Model Integrating Computational Pathology to Predict Early Recurrence of Hepatocellular Carcinoma after Resection.*

__Authors:__ Astrid Laurent-Bellue*, Aymen Sadraoui*, Aurélie Beaufrère, Julien Calderaro, Katia Posseme, Véronique Bruna, Antoinette Lemoine, Agnès Bourillon, Antonio Sa Cunha, Daniel Cherqui, Eric Vibert, Olivier Rosmorduc, Valérie Paradis, Maïté Lewin, Jean-Christophe Pesquet, Catherine Guettier.<br>*These authors contributed equally to this work.
<p align="center">
  <img src="figures/graphical_abstract_UPDATED.jpg"
       alt="Graphical abstract"
       width="1070"
       style="max-width:100%; height:auto;" />
</p>


### Data
<p align="center">
  <img src="figures/Fig4_distribution_of_patients_KbHmBj_UPDATED.jpg"
       alt="data_dist"
       width="600"
       style="max-width:100%; height:auto;" />
</p>

### Results
#### Main cohort: Paul-Brousse
<p align="center">
  <img src="figures/Figure5.jpg"
       alt="Internal_Cohort"
       width="750"
       style="max-width:100%; height:auto;" />
</p>

#### External cohorts: Henri-Mondor & Beaujon
<p align="center">
  <img src="figures/Figure6.jpg"
       alt="external_Cohort"
       width="750"
       style="max-width:100%; height:auto;" />
</p>

### Repository Structure
```
🧬 ML-Model-Integrating-Computational-Pathology-to-Predict-Early-Recurrence-of-HCC
    └── 💾checkpoints
    │    ├── 📁coords_checkpoints
    |    ├── 📁inflam_dats
    |    ├── 📁inflam_checkpoints
    │    └── 📁tumor_checkpoints
    ├── 🗃️data
    │    ├── 📁patches
    │    ├── 📁tabs
    │    └── 📁WSIs
    ├── 🧪experiments
    ├── 🖼️figures
    ├── 🤖models
    │    ├── TripleIndepResNet34_Fold1.pt
    │    ├── TripleIndepResNet34_Fold2.pt
    │    ├── TripleIndepResNet34_Fold3.pt
    │    ├── TripleIndepResNet34_Fold4.pt
    │    └── TripleIndepResNet34_Fold5.pt
    ├── 📓notebooks
    │    ├── color_transfer_from_PB_to_external.ipynb
    │    ├── init_inflam_detection_with_Tia.ipynb
    |    ├── resize_patches.ipynb
    │    ├── STEP1_gen_patches_from_WSI.ipynb
    │    ├── STEP2_detect_tumor_from_WSI.ipynb
    │    └── STEP3_detect_inflammatory_cells.ipynb
    ├── 📊results
    │    ├── 📁overview_preds_inflam_wsis
    │    ├── 📁overview_preds_tumor_wsis
    │    └── 📁overview_wsis
    ├── 📜scripts
    │    ├── run_step1.sh
    │    ├── run_step2.sh
    │    └── run_step3.sh
    ├── 🧩src
    │    ├── STEP0_create_directories.py
    │    ├── STEP1_gen_patches_from_WSI.py
    │    ├── STEP2_detect_tumor_from_WSI.py
    │    └── STEP3_detect_inflammatory_cells.py
    ├── 🛠️utils
    │    ├── ImageSet.py
    │    ├── init.py
    │    ├── model_archi.py
    │    ├── utils_inflams.py
    │    ├── utils_tumor.py
    │    └── utils.py
    ├── .gitignore
    ├── config.yaml
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    └── setup.py
```

### Pretrained Weights (TripleResNet)
The pretrained weights for **TripleResNet34** are hosted externally due to their size.
You can download them from Google Drive:
- **Link:** [TripleResNet pretrained weights](https://drive.google.com/drive/folders/1pLChhs3gZIosXJnp8SyjSAEyNVn88Ffo?usp=drive_link)

After downloading, place the weight files in the appropriate directory `models/`.

> 📄 Note: If you use these pretrained models in your work, please consider citing:
```
@article{LAURENTBELLUE20241684,
title = {Deep Learning Classification and Quantification of Pejorative and Nonpejorative Architectures in Resected Hepatocellular Carcinoma from Digital Histopathologic Images},
journal = {The American Journal of Pathology},
volume = {194},
number = {9},
pages = {1684-1700},
year = {2024},
issn = {0002-9440},
}
```


### Installation
Clone the repo and `cd` into the directory:
```
git clone https://github.com/aymenSadraoui/ML-Model-Integrating-Computational-Pathology-to-Predict-Early-Recurrence-of-HCC.git
cd ML-Model-Integrating-Computational-Pathology-to-Predict-Early-Recurrence-of-HCC
```
Then create a conda env and install the dependencies:
```
conda create -n prognosis python=3.12 -y
conda activate prognosis
pip install -e .
```

### Contact
For any questions or inquiries regarding this project, you can reach me at:  
- **Primary email:** aymen.sadraoui@centralesupelec.fr 
- **Secondary email:** aymen.sadraoui@universite-paris-saclay.fr



<div align="center">
  <img src="figures/centrale_supelec.png" alt="Centrale Supélec" width="270" style="max-width: 100%; height: auto; margin: 0 15px;"/>
  <img src="figures/logo-cvn.png" alt="CVN" width="280" style="background-color: white; max-width: 100%; height: auto; margin: 0 15px;"/>
  <img src="figures/logo_kb.png" alt="bicetre" width="160" style="max-width: 100%; height: auto; margin: 0 15px;"/>
</div>