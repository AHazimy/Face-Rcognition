import os 
import math

BASE = os.path.abspath(os.getcwd())

train_data_path = os.path.join(BASE, "data", "train")
test_data_path = os.path.join(BASE, "data", "test")

for dr in os.listdir(train_data_path):
    print(dr)
    dr = os.path.join(train_data_path, dr)
    if os.path.isdir(dr):
        print("True")
        
        if not os.path.exists(dr.replace(train_data_path, test_data_path)):
            os.makedirs(dr.replace(train_data_path, test_data_path))
        
        _, _, pics = next(os.walk(os.path.join(train_data_path, dr)))
        pics_count = len(pics)
        
        all_pics = os.listdir(dr)
        
        for i in range(math.ceil(pics_count/4)):
            print(i)
            abs_path = os.path.join(dr, all_pics[i])
            os.rename(abs_path, abs_path.replace(f'{train_data_path}', f'{test_data_path}'))