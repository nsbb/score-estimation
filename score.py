#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from torchvision.models.resnet import *
from torchvision.transforms import * 
from loader import coxem_dataset_brcn
from datetime import datetime
import torch.optim as optim
import numpy as np
import argparse
import torch
import time
import sys
import os

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('mode_name')
    parser.add_argument('net_name')
    parser.add_argument('crop_name')
    args = parser.parse_args()

    # path
    data_path = './dataset/' + args.mode_name + '/*/'
    result_path = './result/' + args.mode_name + '_' + args.net_name + '_' + args.crop_name + '.txt'
    trained_path = './trained/' + args.net_name + '_' + args.crop_name + '.pth'

    # hyper-params
    epochs = 100
    batch_size = 1
    lr = 0.00015
    momentum = 0.9
    num_workers = 4

    width = 320
    height = 240

    net = eval(args.net_name+'(img_channel=1, num_classes=2)')

    # preprocess 
    if args.mode_name == 'train':
        if args.crop_name == 'x20':
            transform = Compose([
                        Grayscale(),
                        FiveCrop((height,width)),
                        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
                        ])

        elif args.crop_name == 'x4':
            transform = Compose([
                        Grayscale(),
                        CenterCrop((height,width)),
                        ToTensor()
                        ])

        elif args.crop_name == 'random':
            transform = Compose([
                        Grayscale(),
                        RandomVerticalFlip(),
                        RandomHorizontalFlip(),
                        RandomCrop((height,width)),
                        ToTensor()
                        ])
        else:
            raise ValueError("Crop should be 'x20' or 'random'. Got '{}'".format(args.crop_name))

        print('{} {} started'.format(args.mode_name, args.net_name))

        if os.path.isfile(trained_path):
            while 1:
                reply = input('Trained {} net founded. Would you like to load it? [y/n] : '.format(args.net_name)).lower().strip()
                if reply[0] == 'y':
                    net.load_state_dict(torch.load(trained_path))
                    print('{} parameters loaded.'.format(args.net_name))
                    break
                elif reply[0] == 'n':
                    print('overwrite at trained model.')
                    break
                else: 
                    print('the answer is invalid')

    elif args.mode_name == 'test':
        if os.path.isfile(trained_path):
            transform = Compose([
                        Grayscale(),
                        CenterCrop((height,width)),
                        ToTensor()
                        ]) 
            net.load_state_dict(torch.load(trained_path))
            print('{} {} started'.format(args.mode_name, args.net_name))
            print('{} parameters loaded'.format(args.net_name))
        else:
            raise RuntimeError("Can not load trained model. Check again net_name")

    else:
        raise ValueError("Mode should be 'train' or 'test'. Got '{}'".format(args.mode_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    loss_function = torch.nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    coxem = coxem_dataset_brcn(data_path,
                          transform=transform)

    dataloader = DataLoader(dataset=coxem,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)

    print('{}\n'.format(datetime.now()))
    write_file = open(result_path,'w')
    average_loss = 0.0
    result = []
    img_count = 0

    if args.mode_name == 'train':
        for epoch in range(epochs):
            epoch_start = time.time()
            average_loss = 0.0
            img_count = 0

            for _, (short_name, img, target) in enumerate(dataloader):
                #target_br = target_br.to(device)
                #target_br = target_br.float()
                #target_cn = target_cn.to(device)
                #target_cn = target_cn.float()
                #target = target_br + target_cn
                #print(target_br,target_cn,target)
                target = target.to(device)
                target = target.float()

                img = img.to(device)

                if args.crop_name == 'x4':
                    for flip_control in range(4):
                        if flip_control is 0:
                            img_flip = img
                        elif flip_control is 1:
                            img_flip = torch.flip(img,[3]) # h flip
                        elif flip_control is 2:
                            img_flip = torch.flip(img,[2]) # v flip
                        else:
                            img_flip = torch.flip(img,[2,3]) # hv flip
                        optimizer.zero_grad()

                        output = net(img_flip)
                        loss = loss_function(output.float(), target.float()).float()

                        loss.backward()
                        optimizer.step()

                        average_loss += loss.item()
                        img_count += 1
                        #print(target,output,loss)

                if args.crop_name == 'x20':
                    for crop_control in range(5):
                        img_crop = img[0][crop_control].view(1,1,height,width)
                        for flip_control in range(4):
                            if flip_control is 0:
                                img_flip = img_crop
                            elif flip_control is 1:
                                img_flip = torch.flip(img_crop,[3]) # h flip
                            elif flip_control is 2:
                                img_flip = torch.flip(img_crop,[2]) # v flip
                            else:
                                img_flip = torch.flip(img_crop,[2,3]) # hv flip
                            optimizer.zero_grad()

                            output = net(img_flip)
                            #loss = loss_function(output.view(1).float(), target.view(1).float()).float()
                            loss = loss_function(output.float(), target.float()).float()

                            loss.backward()
                            optimizer.step()

                            average_loss += loss.item()
                            img_count += 1
                            #print(target,output,loss)

                elif args.crop_name == 'random':
                    output = net(img)
                    loss = loss_function(target.float(), output.float()).float()

                    loss.backward()
                    optimizer.step()

                    average_loss += loss.item()    
                    img_count += 1
                    print(target,output,loss)
            epoch_time = time.time()-epoch_start

            print('{:d}, {:5d}, loss: {:.3f}, time: {:.4f}s'.format(epoch+1,img_count,np.sqrt(average_loss/img_count),epoch_time))
            write_file.write('{:d}, {:5d} loss: {:.3f}, time: {:.4f}s\n{}\n\n'.format(epoch+1,img_count,np.sqrt(average_loss/img_count),epoch_time,datetime.now()))
            trained_path = './trained/' + args.net_name + '_' + args.crop_name + '_' + str(epoch+1) + '.pth'
            torch.save(net.state_dict(),trained_path)
            print('{} save done'.format(trained_path))
            print('{}\n'.format(datetime.now()))

    elif args.mode_name == 'test':
        for _, (short_name, img, target) in enumerate(dataloader):
            target = target.to(device)
            target = target.float()

            img = img.to(device)
            with torch.no_grad():
                output = net(img)
                average_loss += loss_function(target.float(), output.float())
                #result.append((short_name[0],target.item(),output.item()))
                #print(short_name[0],target[0][0].item(),target[0][1].item(),output[0][0].item(),output[0][1].item())
                result.append((short_name[0],target[0][0].item(),target[0][1].item(),output[0][0].item(),output[0][1].item()))
                img_count += 1

        list.sort(result)
        for _ in result:
            write_file.write('{}, target: {:.2f} {:.2f}, output: {:.2f} {:.2f}\n'.format(_[0],_[1],_[2],_[3],_[4]))
        write_file.write('RMSE: {}.\n'.format(torch.sqrt(average_loss/(img_count))))

    write_file.write('time = {:.4f}s.\n'.format(time.time()-start))
    write_file.close()
    print('{}_{}_{}.txt file write done'.format(args.mode_name,args.net_name,args.crop_name))
    print(datetime.now()) 

if __name__ == '__main__':
    main()
