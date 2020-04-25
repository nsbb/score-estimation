#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import random
import argparse
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode_name')
    parser.add_argument('folder_name')
    parser.add_argument('par')
    args = parser.parse_args()

    txts_name = [_ for _ in glob.iglob(join('./dataset',args.folder_name,'*.txt'), recursive=True)]
    imgs_name = [_ for _ in glob.iglob(join('./dataset',args.folder_name,'*.jpg'), recursive=True)]

    counter, total, percent, result = [0 for _ in range(10)], [0 for _ in range(10)], [0]*10, []
    max_count = 1000
    ratio = 0.7
    total_count, total_percent = 0, 0

    for txt_name in txts_name:
        read_file = open(txt_name)
        short_name = [txt_name.split('_')[-1]]
        lines = read_file.readlines()
        target_value = [line.split(',')[1].split('\n')[0] for line in lines if 'TARGET_' + args.par in line]
        total[int(target_value[-1])] += 1  

        if args.mode_name == 'check':
            result.append([short_name,target_value])

        elif args.mode_name == 'split':
            result.append([txt_name.split('.txt')[0]+'.jpg',target_value[-1]])

        read_file.close()

    if args.mode_name == 'check':
        name = 'targets.txt'
        write_file = open(name,'w')
        list.sort(result)

        for _ in result:
            #print(_[0],_[1])
            write_file.write('{}, {}\n'.format(_[0][0],_[1][-1]))

        for _ in range(10):
            percent[_] = total[_]/len(imgs_name)*100
            sentence = "'Score {} : {:4d}/{:4d}, {:5.1f}%'.format(_,total[_],len(txts_name),percent[_])"
            print(eval(sentence))
            write_file.write(eval(sentence)+'\n')
            total_count += total[_]
            total_percent += percent[_]

        print('Total D : {:4d}/{:4d}, {:3.1f}%'.format(total_count, len(txts_name), total_percent))
        write_file.write('Total D : {:4d}/{:4d}, {:3.1f}%'.format(total_count, len(txts_name), total_percent))
        write_file.close()

    elif args.mode_name == 'split':
        score_split = [[0 for _ in range(2)] for _ in range(10)]
        total_train, total_test = 0, 0

        for _ in range(10):
            shutil.rmtree('./dataset/train/'+str(_))
            os.makedirs('./dataset/train/'+str(_))
            shutil.rmtree('./dataset/test/'+str(_))
            os.makedirs('./dataset/test/'+str(_))
        print("Flushing 'train' and 'test' folders... Done.\n")
        random.shuffle(result, random.random)

        for _ in result:
            name = str(_[0])
            target = int(_[1])

            if counter[target] < int(max_count*ratio) and counter[target] < int(total[target]*ratio):
                det = './dataset/train/'+str(target)
                #det = './dataset/train/'
                shutil.copy(name, det)
                shutil.copy(name.split('.jpg')[0]+'.txt', det)
                counter[target] += 1
                score_split[target][0] += 1
                total_train += 1

            #elif int(max_count*0.7) <= counter[target] < max_count or (total[target] < max_count and int(total[target]*0.7) <= counter[target]):
            elif (int(max_count*ratio) <= counter[target] or int(total[target]*ratio) <= counter[target]) and counter[target] < max_count:
                det = './dataset/test/'+str(target)
                #det = './dataset/test/'
                shutil.copy(name, det)
                shutil.copy(name.split('.jpg')[0]+'.txt', det)
                counter[target] += 1
                score_split[target][1] += 1
                total_test += 1
            #print('{} moved to {}'.format(name,det))

        for _ in range(10):
            print('Score {} Total: {:4d} Train: {:3d} Test: {:3d}'.format(_,total[_],score_split[_][0],score_split[_][1]))
        
        print('\nTotal Set: {}, Train: {}, Total Test: {}'.format(total_train+total_test, total_train, total_test))
        print('Data Split Done.')
    else:
        raise ValueError("Mode should be 'check' or 'split'. Got'{}'".format(args.mode_name))

if __name__ == '__main__':
    main()
