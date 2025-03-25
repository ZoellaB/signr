import os
import config
import train
    
TRAIN_DIR = "training_data"
GEN_DIR = 'gen_keypoints_data'
START = 0
KEYPTS_PATH = os.path.join(os.curdir, GEN_DIR)

def main():
    words = [i for i in range(2000)]
    to_be_removed = train.get_missing_vids("/missing.txt")
    # tbr = [miss_words.strip() for miss_words in to_be_removed]
    for word_num in words:
        if os.path.isdir(os.path.join(KEYPTS_PATH, str(word_num))):
            folders = os.path.join(KEYPTS_PATH, str(word_num))
            videos = os.listdir(folders)
            for vid in videos:
                if vid in to_be_removed:
                    print("Removing empty files under" + str(word_num))
                    for vid in videos:
                        try: 
                            path = os.path.join(KEYPTS_PATH, str(word_num), str(vid))
                            os.rmdir(path)
                            print(f"Removing {vid} under {word_num}")
                        except Exception as e:
                            print(f"Failed to delete folder: {path}. Error: {e}")
                            pass
                else:
                    pass
        else: 
            print(f"Passing {word_num}")

       

    vidStillInTBR = 0
    for j in range(2000):
        if os.path.isdir(os.path.join(KEYPTS_PATH, str(j))):
            videos = os.listdir(os.path.join(KEYPTS_PATH, str(j)))
            for vid in videos:
                if vid in to_be_removed:
                    vidStillInTBR += 1

    if vidStillInTBR > 0 :
         print("Removed all?: False")
    else:
         print("Removed all?: True")
                    

            
    

    

        


if __name__ == '__main__':
    main()