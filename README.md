# TCA-Net: Triplet Concatenated-Attentional Network for Multimodal Engagement Estimation

This is an official Pytorch implementation of our paper, **TCA-Net: Triplet Concatenated-Attentional Network for Multimodal Engagement Estimation**, to be published in the [IEEE ICIP 2024](https://2024.ieeeicip.org/) conference.

**Abstract:** Human social interactions involve intricate social signals that artificial intelligence and machine learning models aim to decipher, particularly in the context of artificial mediators that can enhance human interactions across domains like education and healthcare. Engagement, a key aspect of these interactions, relies heavily on multimodal information like facial expressions, voice and posture. Recently, many deep learning methods have been deployed in engagement estimation. Still, they often focus on unimodality or bimodality, leading to the results lacking robustness and adaptability due to factors like noise and varying individual responses. To address this challenge, we introduce a novel modality fusion framework named Triplet Concatenated-Attentional Net (_TCA-Net_). This framework takes three distinct types of data modality (video, audio and Kinect) as inputs and delivers a prediction score as output. Within this network, a specially designed concatenated-attention fusion mechanism serves the purpose of modality fusion and preserves the intra-modal features. Experimental results validate the efficiency of our _TCA-Net_ in enhancing the accuracy and reliability of engagement estimation across diverse scenarios, with a test set Concordance Correlation Coefficient (CCC) of 0.75.

**Full paper:** [PDF](https://hasan-rakibul.github.io/pdfs/he2024tca-net.pdf).

--------------------------------------------------------------------------------------------

# Environment

The code is developed using Python 3.8.5. NVIDIA GPUs are needed. The code is developed and tested using one/multiple NVIDIA V100 GPU card. Other platforms or GPU cards are not fully tested.


--------------------------------------------------------------------------------------------

# Quick start

Installation：

Clone this repo

pip install -r requirements.txt

Download the dataset subset: please download and place it under `./data`

Before running, a few available arguments might be adjusted:
    --preprocess: "pca" for doing PCA, else for minmax norm only.
    --data_dim: for pca target dimensions [audio_dim,video_dim,kinect_dim].
    --proj_dim: for projectors output dimensions for aligning multi modalities.

Training
python main.py --method 'TCA_Net' --test False

Testing
python main.py --method 'TCA_Net' --test False


Furthermore, as introduced in the ablation study, if users would like to reproduce engagement estimation training/testing based on dual-modality, the `args.method` should be modified to "DCA_Net". 

--------------------------------------------------------------------------------------------

# If you find this repository useful, please cite our paper
```bibtex
@InProceedings{he2024tca-net,
    author = {He, Hongyuan and Wang, Daming and Hasan\supervisor, Md Rakibul and Gedeon, Tom and Hossain, Md Zakir},
    title = {{TCA}-{N}et: Triplet Concatenated-Attentional Network for Multimodal Engagement Estimation},
    booktitle = {2024 IEEE International Conference on Image Processing (\textbf{ICIP})},
    year = {2024},
    organization={IEEE}
}
```
