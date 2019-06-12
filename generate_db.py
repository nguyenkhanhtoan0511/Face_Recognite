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

def generate_data(src, db):
    train_files = []
    test_files = []
    for folder in os.listdir(src):
        print("[+]Access folder ",folder)
        folder_path = os.path.join(src, folder)        
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        n = len(files)
        n_train = int(n * 0.7)
        train_files.extend(generate_train(files, n_train))
        test_files.extend(generate_test(files, train_files))
    print("[+]Create folder ", db)
    os.makedirs(db)
    print("[+]Change current wd to ", db)
    os.chdir(db)
    write_file("train.txt", train_files)
    write_file("test.txt", test_files)

def main():
    src = sys.argv[1] # path file Images
    db = sys.argv[2] # folder that will be created
    generate_data(src, db)

if __name__=='__main__':
    main()