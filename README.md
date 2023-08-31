# TCA-Net: Triplet Concatenated-Attentional Network for Multimodal Engagement Estimation

This is an official pytorch implementation of TCA-Net: Triplet Concatenated-Attentional Network for Multimodal Engagement Estimation.

To address this Multimodal Engagement Estimation, this work introduces a novel modality fusion framework -- TCA-Net.  This framework takes three distinct types of modality data (video, audio and Kinect) as inputs and delivers a prediction score as output.  Within this network, a specially designed concatenated-attention fusion mechanism not only serves the purpose of modality fusion but also preserves the intra-modal features.

--------------------------------------------------------------------------------------------

Environment

The code is developed using python 3.8.5. NVIDIA GPUs are needed. The code is developed and tested using one/multiple NVIDIA V100 GPU card. Other platforms or GPU cards are not fully tested.


--------------------------------------------------------------------------------------------

Quick start

Installation：
Clone this repo
pip install -r requirements.txt
Download dataset subset, please download and place under /data

Before the runing, there are few availabel arguments might be adjusted:
    --preprocess: "pca" for doing PCA, else for minmax norm only.
    --data_dim: for pca target dimensions [audio_dim,video_dim,kinect_dim].
    --proj_dim: for projectors output dimensions for aligning multi modalities.

Training
python main.py --method 'TCA_Net' --test False

Testing
python main.py --method 'TCA_Net' --test False


Furthermore, as introduced in the ablation study, if users would like to reproduce engagement estimation training/testing based on dual-modality. The args.method could be modified to "DCA_Net". 

--------------------------------------------------------------------------------------------
