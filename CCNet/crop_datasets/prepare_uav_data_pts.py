import argparse
import os
import shutil
import json
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training/testing data generation', add_help=False)
    parser.add_argument('--ratio_training', default=0.9, type=float,
                        help="The ration of training data")
    parser.add_argument('--data_path', default='./Rice_UAV_dataset_10m', type=str,
                        help="The path of original data")
    return parser

def points_from_json(json_file):
    DataJson = json.load(open(json_file))
    shpae_file = DataJson["shapes"]
    points_data = []
    for points in shpae_file:
        point = points["points"][0]
        points_data.append(point)
        #point= np.array([[point_x, point_y]])
        #points_data = np.concatenate((points_data, point), axis=0)
    points_data = np.array(points_data)
    #print(points_data)
    return points_data

def prepare_training_testing_img_txt():
    file_count_total = 0
    file_count_train = 1
    file_count_test = 1
    fp_train = open("./part_a_train.list", "w")
    fp_test = open("./part_a_test.list", "w")
    for filepath, dirnames, filenames in os.walk(args.data_path+"/img"):
        for filename in filenames:
            file_img_path=filepath+"/"+filename
            print(file_img_path)
            file_json_path = (file_img_path.replace("/img", "/json")).replace(".jpg", ".json")
            print(file_json_path)
            points = points_from_json(file_json_path)
            #print(points)
            print(len(filenames))
            if(file_count_total < args.ratio_training*len(filenames)):
                train_file = "./train/scene01/"+"img_"+str(file_count_train)+".jpg"
                shutil.copyfile(file_img_path, train_file)
                print(train_file)
                train_txt = train_file.replace(".jpg", ".txt")
                np.savetxt(train_txt, points)
                fp_train.write("%s %s\n"%(train_file.replace("./train", "train"), train_txt.replace("./train", "train")))
                #print("train")
                file_count_train += 1
            else:
                test_file = "./test/scene01/" + "img_" + str(file_count_test) + ".jpg"
                shutil.copyfile(file_img_path, test_file)
                print(test_file)
                test_txt = test_file.replace(".jpg", ".txt")
                np.savetxt(test_txt, points)
                fp_test.write("%s %s\n"%(test_file.replace("./test", "test"), test_txt.replace("./test", "test")))
                #print("test")
                file_count_test += 1
            file_count_total += 1

    fp_train.close()
    fp_test.close()



def main(args):
    # backup the arguments
    print(args)

    # Prepare the training and testing data
    prepare_training_testing_img_txt()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare UAV data for Crop Counting', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)




