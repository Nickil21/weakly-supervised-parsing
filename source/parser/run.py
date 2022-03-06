from source.utils.align_ptb import AlignPTBYoonKimFormat
from source.utils.prepare_dataset import DataLoader

from source.settings import PTB_TRAIN_GOLD_PATH, PTB_VALID_GOLD_PATH, PTB_TEST_GOLD_PATH
from source.settings import PTB_TRAIN_GOLD_ALIGNED_PATH, PTB_VALID_GOLD_ALIGNED_PATH, PTB_TEST_GOLD_ALIGNED_PATH
from source.settings import YOON_KIM_TRAIN_GOLD_PATH, YOON_KIM_VALID_GOLD_PATH, YOON_KIM_TEST_GOLD_PATH

# Align PTB with YK
AlignPTBYoonKimFormat(ptb_data_path=PTB_TRAIN_GOLD_PATH, yk_data_path=YOON_KIM_TRAIN_GOLD_PATH).row_mapper(save_data_path=PTB_TRAIN_GOLD_ALIGNED_PATH)
AlignPTBYoonKimFormat(ptb_data_path=PTB_VALID_GOLD_PATH, yk_data_path=YOON_KIM_VALID_GOLD_PATH).row_mapper(save_data_path=PTB_VALID_GOLD_ALIGNED_PATH)
AlignPTBYoonKimFormat(ptb_data_path=PTB_TEST_GOLD_PATH, yk_data_path=YOON_KIM_TEST_GOLD_PATH).row_mapper(save_data_path=PTB_TEST_GOLD_ALIGNED_PATH)


