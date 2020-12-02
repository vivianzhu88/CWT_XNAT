import torch.utils.data as data
import torch
import numpy as np
import sys

from PIL import Image
import os
import os.path
from torchvision import transforms
from utils.convert_dicom_png import process_dicom
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.mat', '.npy'
]

import xnat
import os
session = xnat.connect('http://rufus.stanford.edu', user='admin', password='admin') #make XNAT connection


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid data directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images



def make_dataset_dicom(dir, opt, phase):
    images = []
    assert os.path.isdir(dir), '%s is not a valid data directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.dcm'):
                try:
                    opt.names_phase[fname]
                except:
                    print('Error: we donot find the phase for file name' , fname, 'ignore this file')
                    continue
                    # sys.exit(3)
                if opt.names_phase[fname] == phase:
                    path = os.path.join(root, fname)
                    Img = process_dicom(path)
                    if Img is not None:
                        images.append(path)

    # print('We use dicom for train--updated')
    return images


def make_dataset_XNAT(project, opt, phase):
    #xnat version of make_dataset_dicom()
    images = []
    #parsing xnat hierarchy: project -> subject -> experiment -> scan -> dicom img
    subjs = session.projects[project].subjects
    for s in subjs:
        exps = subjs[s].experiments
        
        for e in exps:
            scans = exps[e].scans
            
            for sc in scans:
                my_file = scans[sc].resources['DICOM'].files[0]
                fname = my_file.data['Name']

                #save images whose filenames are in phase
                try:
                    opt.names_phase[fname]
                except:
                    print('Error: we donot find the phase for file name' , fname, 'ignore this file')
                    continue
                    # sys.exit(3)
                if opt.names_phase[fname] == phase:
                    Img = process_dicom(my_file)
                    if Img is not None:
                        images.append([fname, my_file]) #each image list item has filename + FileData obj

    return images


class DatasetDist(data.Dataset):
    def __init__(self, opt, phase):
        super(DatasetDist, self).__init__()
        if opt.xnat_proj_id == '':
            #self.img_paths = make_dataset_dicom(os.path.join(opt.data_path, folder))
            self.img_paths = make_dataset_dicom(opt.data_path, opt, phase)
        else:
            self.img_paths = make_dataset_XNAT(opt.xnat_proj_id, opt, phase)
            self.is_xnat = True
        print('Loading', len(self.img_paths), 'training images from institution', opt.inst_id, 'for', phase, '------')
        self.opt = opt
        self.labels = opt.labels

        if opt.phase == 'train':
            data_transforms = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=0.5),
                 transforms.Resize([opt.load_size, opt.load_size]),
                 transforms.RandomCrop([opt.fine_size, opt.fine_size]),
                 transforms.ToTensor(),
                 # transforms.Normalize([0.485], [0.229])
                ])
        else:
            data_transforms = transforms.Compose(
                [transforms.Resize([opt.load_size, opt.load_size]),
                 transforms.CenterCrop([opt.fine_size, opt.fine_size]),
                 transforms.ToTensor(),
                 # transforms.Normalize([0.485], [0.229])
                 ])

        self.transform = data_transforms

    def __getitem__(self, index):
        if self.is_xnat:
            name = self.img_paths[index][0] #the name
            my_file = self.img_paths[index][1] #fileData obj
        else:
            name = self.img_paths[index] #local file path
            my_file = self.img_paths[index] #local file path

        if name.endswith('.npy'):
            Img = np.load(my_file)
            Img = (Img - Img.min()) / (Img.max() - Img.min()) * 224
            Img = Image.fromarray(Img.astype('uint8'))
        elif name.endswith('.dcm'):
            Img = process_dicom(my_file)
            Img = Img * 255
            Img = Image.fromarray(Img.astype('uint8'))

        else:
            Img = Image.open(my_file).convert('RGB')

        input = self.transform(Img)
        if input.shape[0] == 1:
            input = torch.cat([input, input, input])
        tmp_label = self.labels[name]

        if self.opt.regression:
            label = torch.FloatTensor(1)
            label[0] = tmp_label
        else:

            label = torch.LongTensor(3)

            label[0] = tmp_label
            label[1] = tmp_label
            label[2] = tmp_label
        return {'input': input, 'label': label, 'Img_paths': name}

    def __len__(self):
        return len(self.img_paths)



