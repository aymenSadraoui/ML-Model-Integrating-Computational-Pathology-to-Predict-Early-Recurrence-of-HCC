# Machine Learning Model Integrating Computational Pathology to Predict Early Recurrence of Hepatocellular Carcinoma after Resection

This repository contains the code and supplementary materials for our article:
*Machine Learning Model Integrating Computational Pathology to Predict Early Recurrence of Hepatocellular Carcinoma after Resection*

__Authors:__ Astrid Laurent-Bellue*, Aymen Sadraoui*, Aurélie Beaufrère, Julien Calderaro, Katia Posseme, Véronique Bruna, Antoinette Lemoine, Agnès Bourillon, Antonio Sa Cunha, Daniel Cherqui, Eric Vibert, Olivier Rosmorduc, Valérie Paradis, Maïté Lewin, Jean-Christophe Pesquet, Catherine Guettier
*These authors contributed equally to this work
![Graphical abstract](figures/graphical_abstract_UPDATED.jpg)  


## Repository Structure
```
📁ML-Model-Integrating-Computational-Pathology-to-Predict-Early-Recurrence-of-HCC
    └── 📁checkpoints
        └── 📁coords_pickles
    └── 📁data
        └── 📁patches
        └── 📁tabs
            ├── table_prognosis.xlsx
        └── 📁WSIs
            └── 📁BJ
                └── 📁Patient_161
                    ├── 161A.svs
                    ├── 161B.svs
            └── 📁HM
                └── 📁Patient_111
                    ├── 111A.csv
                    ├── 111A.ndpi
                    ├── 111A.ndpi.ndpa
                    ├── 111B.ndpi
                    ├── 111C.ndpi
            └── 📁PB
                └── 📁Patient_1
                    └── 📁1A
                    └── 📁1B
                    └── 📁1C
                    ├── 1A_Annotations.xml
                    ├── 1A.mrxs
                    ├── 1B_Annotations.xml
                    ├── 1B.mrxs
                    ├── 1C_Annotations.xml
                    ├── 1C.mrxs
    └── 📁experiments
    └── 📁figures
    └── 📁notebooks
        ├── STEP1&1bis_gen_patches_from_WSI.ipynb
    └── 📁results
    └── 📁src
        ├── __init__.py
    └── 📁utils
        ├── __init__.py
        ├── utils.py
    ├── .gitignore
    ├── init.py
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    └── setup.py
```

## Data
![data_dist](figures/Fig4_distribution_of_patients_KbHmBj_UPDATED.jpg)

## Results
### Main cohort: Paul-Brousse
![Internal_Cohort](figures/Figure5.jpg)  

### External cohorts: Henri-Mondor & Beaujon
![Internal_Cohort](figures/Figure6.jpg) 