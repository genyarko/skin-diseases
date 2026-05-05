# skin-diseases
# 1. Kaggle credentials
pip install kaggle
mkdir -p ~/.kaggle && \
  echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json && \
  chmod 600 ~/.kaggle/kaggle.json

# 2. Clone repo
git clone https://github.com/genyarko/skin-diseases
cd skin-diseases

apt install python3.12-venv
python3 -m venv ~/venv

# 3. Environment (ROCm-aware venv)
bash setup_rocm.sh
source ~/venv/bin/activate
pip3 install -r requirements.txt
pip3 install timm scikit-learn kaggle
pip3 install opencv-python  # only if you set APPLY_HAIR_REMOVAL=True

python pretrain_ham10000.py
python train_amd.py

Step 1 — Quick win: re-eval current model with proper TTA (~3 min)
  python ensemble_eval.py best_ema.pt                                                                                                                            
  Should recover the 0.6pp lost to the broken pixel-roll TTA. Likely lands at ~80.5-81.0%.                                                                     
                                                                                                                                                                 
  Step 2 — Continuation training (~1 hour)
  python continue_train.py
  Loads best_ema.pt fully, runs 15 epochs at LR=1e-5 with mixup off, EMA + SWA throughout. Target: ~82-83%. Outputs best_ema_v2.pt and swa_v2.pt.

  Step 3 — Train ConvNeXt V2-Huge as ensemble partner (~2 hours)
  python train_convnextv2.py
  Different architecture (CNN, not ViT) means independent error patterns — that's where ensemble gain comes from. Expected standalone: ~76-79%. Outputs
  convnextv2_best_ema.pt (and SWA).

  Step 4 — Final ensemble (~5 min)
  python ensemble_eval.py best_ema_v2.pt swa_v2.pt convnextv2_best_ema.pt convnextv2_swa.pt
  Averages softmax across all 4 checkpoints with 4-aug TTA each (identity + hflip × 2 crop scales = 16 total forward passes per image).