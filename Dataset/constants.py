import logging
log: logging.Logger
##################
DATA_PREPROCESSING = 1 ### For the first time you need to set `DATA_PREPROCESSING=1`
IMAGE_SIZE = 64         ### 64 for 64x64 or 32 for 32x32
RANDOM_SEED = 0 ### The random seed for reproducibility.
################### When using UTK dataset, please uncomment this part and comment out the CelebA part below
DATASET = "utk_face"    ### Total images are 23705
TRAIN_SHARE = 18964     ### Then the Test_Share is (23705 - TRAIN_SHARE)
VALID_SHARE = 1896      ### The portion used during training as validation set.
AGE = 0    # DON'T Change this
GENDER = 1 # DON'T Change this
RACE = 2   # DON'T Change this
HONEST = AGE  # This is the target attribut (i.e., y) in the paper. Use this to set your desired attribute.
CURIOUS =  RACE # This is the sensitive attribut (i.e., s) in the paper. Use this to set your desired attribute.
K_Y = 5  ##Set this based on the chosen attribute. For AGE and RACE from 2 to 5, for GENDER only 2.
K_S = 5  ##Set this based on the chosen attribute. For AGE and RACE from 2 to 5, for GENDER only 2.
MAX_Y = 4
MAX_S = 4

################## When using Celeba dataset, please uncomment this part and comment out the UTK part above
## For 'Hair_Color' attribute use code 0
##  For other attributes chooes the code from the following list:
##  {'Attractive': 2, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
## 'Male': 20, 'Mouth_Slightly_Open': 21, 'Smiling': 31, 'Wavy_Hair': 33, 'Wearing_Lipstick': 36}
# DATASET = "celeba"
# HONEST = 0 # This is the target attribut (i.e., y) in the paper. Use this to set your desired attribute.
# CURIOUS = 31  # This is the sensitive attribut (i.., s) in the paper. Use this to set your desired attribute.
# K_Y = 3  ##Set this based on the chosen attribute. For Hair_Color 3, for the rest 2.
# K_S = 2  ##Set this based on the chosen attribute. For Hair_Color 3, for the rest 2.

##################
BETA_X = 2. ## No specific range, but better to set something between 0. to 10.
BETA_Y = 5. ## No specific range, but better to set something between 0. to 10.
BETA_S = 5. ## No specific range, but better to set something between 0. to 10.
##################
# For RawHBC, set to False. For SoftHBC, set to True
SOFTMAX = True
##################
## Optional: for regularized attacks this will be used to report accuracy on the validation set.
REGTAU = .25
##################
summary_file = "summary.csv"
result_path = "/results/par/"
fair_epsilon = 0.2
fair_delta = 0.1
dp_epsilon = 1