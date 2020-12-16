import datetime


DATA_NAME = 'art'

DATALOADER_WORKERS = 8
NBR_CLS = 500

EPOCH_GAN = 20

SAVE_IMAGE_INTERVAL = 500
SAVE_MODEL_INTERVAL = 2000
LOG_INTERVAL = 100
FID_INTERVAL = 2000
FID_BATCH_NBR = 500

ITERATION_AE = 100000

NFC=64
MULTI_GPU = False


IM_SIZE_GAN = 1024
BATCH_SIZE_GAN = 16

IM_SIZE_AE = 512
BATCH_SIZE_AE = 32

ct = datetime.datetime.now()  
TRIAL_NAME = 'trial-pr-%s-%d-%d-%d-%d'%(DATA_NAME, ct.month, ct.day, ct.hour, ct.minute)
SAVE_FOLDER = './'

PRETRAINED_AE_PATH = './train_results/AE_trial-pr-art-8-11-1-35' #None
PRETRAINED_AE_ITER = 12000

GAN_CKECKPOINT =None

TRAIN_AE_ONLY = False
TRAIN_GAN_ONLY = False

#data_root_colorful = './shoe/shoe_rgb/'
#data_root_sketch_1 = './shoe/shoe_skt_1'
#data_root_sketch_2 = './shoe/shoe_skt_2'
#data_root_sketch_3 = './shoe/shoe_skt_3'

#data_root_colorful = '../images/art_landscape/image_512/img'
#data_root_sketch_1 = './sketch_simplification/vggadin_art-landscape_iter_1400'
#data_root_sketch_2 = './sketch_simplification/vggadin_art-landscape_iter_2600'
#data_root_sketch_3 = './sketch_simplification/vggadin_art-landscape_iter_2800'

#data_root_colorful = '../images/art_landscape/image_512/img'
#data_root_sketch_1 = './sketch_simplification/vggadin_art-landscape_iter_1400'
#data_root_sketch_2 = './sketch_simplification/vggadin_art-landscape_iter_2600'
#data_root_sketch_3 = './sketch_simplification/vggadin_art-landscape_iter_2800'

data_root_colorful = '../images/photo-realistic-landscape/img'
data_root_sketch_1 = './model_vggadain/pr-landscape_sketch_iter_3000'
data_root_sketch_2 = './model_vggadain/pr-landscape_sketch_iter_4000'
data_root_sketch_3 = './model_vggadain/pr-landscape_sketch_iter_5000'