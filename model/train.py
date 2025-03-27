import keras.src.utils.numerical_utils as utils
from sklearn.model_selection import train_test_split
import os
import config
import get_keypts

KEYPTS_PATH = config.KEYPTS_PATH





def main():
    max_word_num = len(os.listdir(KEYPTS_PATH))
    label_map = get_keypts.get_word_classification("/wlasl_class_list.txt")
    
    print(KEYPTS_PATH)
    print(max_word_num)
    print(label_map)


if __name__ == '__main__':
    main()