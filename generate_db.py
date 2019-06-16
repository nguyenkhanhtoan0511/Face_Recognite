import sys
import os
import random

def generate_train(files, n):
    random.shuffle(files)
    return random.sample(files, n)    

def generate_test(files, train_files):
    return list(set(files) - set(train_files))

def write_file(path, files):
    print("[+]Write ", path)
    with open(path, "w") as f:
        for file in files:
            f.write(file)
            f.write("\n")            

def generate_data(src1, src2, db):
    train_files = []
    test_files = []
    print(os.listdir(src1))
    #generate train.txt
    for folder in os.listdir(src1):
        print("[+]Access folder ",folder)
        folder_path = os.path.join(src1, folder)
        print('folder_path: ',folder_path)        
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        n = len(files)
        train_files.extend(generate_train(files, n))
    if(os.path.exists(db)==False):
        os.makedirs(db)
    print("[+]Change current wd to ", db)
    os.chdir(db)
    write_file("train.txt", train_files)
    os.chdir("..")
    print(os.listdir(src2))
    # generate test.txt
    for folder2 in os.listdir(src2):
        print("[+]Access folder ",folder2)
        folder_path2 = os.path.join(src2, folder2)
        files = [os.path.join(folder_path2, file) for file in os.listdir(folder_path2)]
        n = len(files)
        test_files.extend(generate_train(files, n))
    if(os.path.exists(db)==False):
        os.makedirs(db)
    print("[+]Change current wd to ", db)
    os.chdir(db)
    write_file("test.txt", test_files)

def main():
    src1 = "./dataset/train_set" # path file Images train
    src2 = "./dataset/test_set" # path file Images test
    db = "./db" # folder that will be save
    generate_data(src1, src2, db)

if __name__=='__main__':
    main()