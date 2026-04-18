# Multi-view-clinical-note-imputation
This repo contains the code for missingness-informed multi-view imputation of clinical notes for health risk prediction.

1-Install the requirements using:
~~~
conda env create -f requirements.yml -n newname
~~~
2-Since the dataset is restricted you have to download it from [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/).

3-You have to preprocess NOTEEVENTS.csv using the preprocessing method in [Link text](https://github.com/LeiGong0125Carrot/Strucure-Awared-Clinical-Note-Processing/tree/ICDM-2025).
4-There is no parsing function in the code for the current version, so you have to change the hyperparameters within the script. Next updates will make the code more modular and user-friendly.
