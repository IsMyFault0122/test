import argparse

import torch
import torchvision.transforms as standard_transforms# type:ignore
import numpy as np

from PIL import Image# type:ignore
import cv2# type: ignore
from .engine import *
from .models import build_model
import os
from .util.nms import nms, nms_array
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for CCNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--input_dir', default='',
                        help='path where to find input images')
    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--block_size_x', default=1700, type=int,
                        help="the width of block without overlapping")
    parser.add_argument('--block_size_y', default=1800, type=int,
                        help="the height of block without overlapping")

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def test_single_img(img_path, img_output, transform, device, model):
    # set your image path here
    #img_path = "./vis/clip_25_10.jpg"
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    ratio_width = new_width/width
    ratio_height = new_height/height
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.5
    # filter the predictions
    filterd_scores = outputs_scores[outputs_scores > threshold]
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    if(1<=len(points)):
        filterd_points = nms(points, filterd_scores, 20.0)
    else:
        filterd_points = np.empty(shape=[0, 2])

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 20
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    # Write the number of points to images
    text = str(len(filterd_points))
    img_to_draw = cv2.putText(img_to_draw, text, (50, 50), cv2.cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    fp_points = open(img_output+".txt", "w")

    for p in filterd_points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), 2)
        fp_points.write("%d %d\n"%(int(p[0]), int(p[1])))
        # img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    # cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)
    output_img_path = img_output+"___"+ str(filterd_points.shape[0])+".jpg"
    cv2.imwrite(output_img_path, img_to_draw)
    fp_points.close()
    return ratio_width, ratio_height, filterd_points


def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_plants=0
    total_det_pts = np.empty(shape=[0, 2])
    for filepath, dirnames, filenames in os.walk(args.input_dir):
        for filename in filenames:
            file_img_path = filepath+filename
            out_img_path = args.output_dir+filename.replace(".jpg", "")
            ratio_w, ratio_h, filted_det_pts = test_single_img(file_img_path, out_img_path, transform, device, model)
            row_id = int(filename.split('.')[0].split('clip_')[-1].split('_')[0])
            col_id = int(filename.split('.')[0].split('clip_')[-1].split('_')[-1])

            #print("%d %d"%(row_id, col_id))
            filted_det_pts[:,0] = filted_det_pts[:,0]/ratio_w + row_id*args.block_size_x
            filted_det_pts[:,1] = filted_det_pts[:,1]/ratio_h + col_id*args.block_size_y
            total_det_pts = np.concatenate((total_det_pts, filted_det_pts), axis=0)
            print(out_img_path)

    all_scores = np.zeros(total_det_pts.shape[0], np.float32)
    all_scores.fill(0.6)
    total_det_pts = nms_array(total_det_pts, all_scores, 20.0)
    np.savetxt(args.output_dir+"detected_pts.txt", total_det_pts)
    print("All images processed!!! There are %d"%(total_det_pts.shape[0]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('CCNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)