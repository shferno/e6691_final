# e6691-2022Spring-project-ccsz-cz2678-yc3998-fs2752

## Prediction of Remaining Surgical Duration

## Description: 
This project reproduced the basic structure and workflow of CataNet for the prediction of remaining surgical durations (RSD) proposed by the original paper. 
The model was also implemented on the surgery videos of hernia reduction to test the generalization ability. 
It can be observed from the experiments that while a CNN+LSTM structure performs well for RSD prediction, surgical phase segmentation can also be obtained as a reasonable auxiliary information of the model.

- [Model link](https://drive.google.com/file/d/1EywxhY6YPZ7a5I8pN5m3_bcuskym6RtC/view?usp=sharing)

## Data Preparation

In order to preprocess the data, please install [ffmpeg library](https://ffmpeg.org/download.html).

Other requirements please run:
```shell
$ pip install requirements.txt
```

## Training

To get the same result, please download the [cataract-101](http://ftp.itec.aau.at/datasets/ovid/cat-101/) dataset, modified [label files](https://zenodo.org/record/4984167#.Ynsb35PMK3I) and [reduction hernia](https://drive.google.com/drive/folders/1yhR6fSOW0gyJyZ-Yy8dnDdUyI3IBFAIl). 

Then run:
```shell
$ python main.py
```

Or: 
- Model structure definition: model/svrc.py - SVRC
- Trainer for Cataract dataset: utils/trainer.py - Trainer
- CNN trainer for Hernia dataset: utils/trainer.py - ResnetTrainer
- RNN trainer for Hernia dataset: utils/trainer.py - LSTMTrainer

## Evaluation

To get the result of reduction hernia visualize, please run

```shell
$ python inference_hernia.py --out 'path to output folder' --input 'path to processed video file or multiple files' --checkpoint 
'path to model checkpoint .pth file'
```

## Directory Organization

```
.
├── LICENSE
├── README.md
├── config.py
├── data
│   ├── images
│   ├── videos
│   ├── cataract101
│   └── labels
│       ├── all_labels_hernia.csv
│       ├── kaggle_template.csv
│       └── video.phase.trainingData.clean.StudentVersion.csv
├── inference_hernia.py
├── main.ipynb
├── main.py
├── models
│   ├── CNN_model.py
│   ├── RNN_model.py
│   └── svrc.py
├── preprocess.ipynb
├── requirements.txt
├── tree.txt
└── utils
    ├── dataset.py
    ├── logging_utils.py
    ├── metrics.py
    ├── prepare.py
    ├── trainer.py
    └── trainer_svrc.py

4 directories, 21 files
```

## Reference

Marafioti, A., Hayoz, M., Gallardo, M., Márquez Neila, P., Wolf, S., Zinkernagel, M., & Sznitman, R. (2021, September). CataNet: Predicting remaining cataract surgery duration. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 426-435). Springer, Cham.

Jin, Y., Dou, Q., Chen, H., Yu, L., Qin, J., Fu, C. W., & Heng, P. A. (2017). SV-RCNet: workflow recognition from surgical videos using recurrent convolutional network. IEEE transactions on medical imaging, 37(5), 1114-1126.

K. Schoeffmann, M. Taschwer, S. Sarny, B. Mu ̈nzer, M. J. Primus, and D. Putz- gruber, “Cataract-101 - Video dataset of 101 cataract surgeries,” Proceedings of the 9th ACM Multimedia Systems Conference, MMSys 2018, pp. 421–425, 2018.

A.Achiron,F.Haddad,M.Gerra,E.Bartov,andZ.Burgansky-Eliash,“Predicting cataract surgery time based on preoperative risk assessment,” European Journal of Ophthalmology, vol. 26, no. 3, 2016.

S. P. Devi, K. S. Rao, and S. S. Sangeetha, “Prediction of surgery times and scheduling of operation theaters in ophthalmology department,” Journal of Medical Systems, vol. 36, no. 2, pp. 415–430, 2012.

M. Lanza, R. Koprowski, R. Boccia, K. Krysik, S. Sbordone, A. Tartaglione, A. Ruggiero, and F. Simonelli, “Application of artificial intelligence in the anal- ysis of features affecting cataract surgery complications in a teaching hospital,” Frontiers in Medicine, vol. 7, 2020.

N. Padoy, T. Blum, H. Feussner, M. O. Berger, and N. Navab, “On-line recognition of surgical activity for monitoring in the operating room,” in Proceedings of the National Conference on Artificial Intelligence, vol. 3, 2008.

S. Franke, J. Meixensberger, and T. Neumuth, “Intervention time prediction from surgical low-level tasks,” Journal of Biomedical Informatics, vol. 46, no. 1, 2013.

A. C. Gu ́edon, M. Paalvast, F. C. Meeuwsen, D. M. Tax, A. P. van Dijke, L. S. Wauben, M. van der Elst, J. Dankelman, and J. J. van den Dobbelsteen, “‘It is Time to Prepare the Next patient’ real-time prediction of procedure duration in laparoscopic cholecystectomies,” Journal of Medical Systems, vol. 40, no. 12, 2016.

N. Spangenberg, M. Wilke, and B. Franczyk, “A big data architecture for intra- surgical remaining time predictions,” in Procedia Computer Science, vol. 113, 2017.

M. Maktabi and T. Neumuth, “Online time and resource management based on surgical workflow time series analysis,” International Journal of Computer Assisted Radiology and Surgery, vol. 12, no. 2, 2017.

S. Bodenstedt, M. Wagner, L. Mu ̈ndermann, H. Kenngott, B. Mu ̈ller-Stich, M. Breucha, S. T. Mees, J. Weitz, and S. Speidel, “Prediction of laparoscopic pro- cedure duration using unlabeled, multimodalsensor data,” International Journal of Computer Assisted Radiology and Surgery, vol. 14, no. 6, 2019.

I. Aksamentov, A. P. Twinanda, D. Mutter, J. Marescaux, and N. Padoy, “Deep neural networks predict remaining surgery duration from cholecystectomy videos,” in Medical Image Computing and Computer-Assisted Intervention - MICCAI 2017, M. Descoteaux, L. Maier-Hein, A. Franz, P. Jannin, D. L. Collins, and S. Duchesne,Eds. Cham: Springer International Publishing, 2017, pp. 586–593.

