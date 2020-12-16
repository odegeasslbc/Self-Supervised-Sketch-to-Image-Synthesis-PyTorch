from train_step_1_ae import train as train_ae
from train_step_2_gan import train as train_gan






if __name__ == "__main__":
    
    from config import TRAIN_GAN_ONLY, TRAIN_AE_ONLY
    if TRAIN_GAN_ONLY:
        train_gan()
    else:
        train_ae()
        if not TRAIN_AE_ONLY:
            train_gan()