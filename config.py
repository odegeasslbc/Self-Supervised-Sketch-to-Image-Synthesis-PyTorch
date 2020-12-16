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

PRETRAINED_AE_PATH = 'add/the/pre-trained/model/path/if/fintuning' #None
PRETRAINED_AE_ITER = 12000

GAN_CKECKPOINT =None

TRAIN_AE_ONLY = False
TRAIN_GAN_ONLY = False

data_root_colorful = '/path/to/image/folder'
data_root_sketch_1 = '/path/to/sketch/folder'
data_root_sketch_2 = '/path/to/sketch/folder'
data_root_sketch_3 = '/path/to/sketch/folder'
