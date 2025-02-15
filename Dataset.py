import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os
import nibabel as nib

class TrainDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir,data_reverse_dir,target_dir,mask_dir,replace_mask_dir,
                 changetable_dir,changetable_reverse_dir,data_type='float32', transform=None):
        self.transform = transform
        self.data_type = data_type
        self.data = np.load(data_dir)
        self.data_reverse = np.load(data_reverse_dir)
        self.target = np.load(target_dir)
        self.mask = np.load(mask_dir)
        self.replace_mask=np.load(replace_mask_dir)
        self.changetable=np.load(changetable_dir)
        self.changetable_reverse=np.load(changetable_reverse_dir)


    def __getitem__(self, index):
        data_image=self.data[..., index]
        data_reverse_image = self.data_reverse[..., index]
        target_image = self.target[..., index]
        mask_image=self.mask[..., index]
        replace_mask_image=self.replace_mask[..., index]
        changetable_image=self.changetable[...,index]
        changetable_reverse_image=self.changetable_reverse[...,index]
        data = {'input': data_image,'input_reverse':data_reverse_image,'target':target_image,'mask':mask_image,
                'replace_mask':replace_mask_image,'change':changetable_image,'change_reverse':changetable_reverse_image}
        data = self.transform(data)
        return data

    def __len__(self):
        return self.data.shape[-1]

class ValDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir,mask_dir,label_dir,data_type='float32', transform=None):
        self.transform = transform
        self.data_type = data_type
        self.data = np.load(data_dir)
        self.mask = np.load(mask_dir)


    def __getitem__(self, index):
        data_image=self.data[..., index]
        mask_image=self.mask[..., index]
        data = {'input': data_image, 'mask':mask_image}
        data = self.transform(data)
        return data

    def __len__(self):
        return self.data.shape[-1]


class TrainToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):

        input, input_reverse, target, mask,replace_mask,changetable,changetable_reverse = data['input'], data['input_reverse'], data['target'],data['mask'],data['replace_mask'],data['change'],data['change_reverse']
        input = input.transpose((2, 0, 1)).astype(np.float32)
        input_reverse = input_reverse.transpose((2, 0, 1)).astype(np.float32)
        target = target.transpose((2, 0, 1)).astype(np.float32)
        mask=mask.transpose((2, 0, 1)).astype(np.float32)
        replace_mask = replace_mask.transpose((2, 0, 1)).astype(np.float32)
        changetable=changetable.transpose((2,0,1)).astype(np.float32)
        changetable_reverse=changetable_reverse.transpose((2,0,1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'input_reverse':torch.from_numpy(input_reverse),
                'target': torch.from_numpy(target), 'mask': torch.from_numpy(mask),
                'replace_mask':torch.from_numpy(replace_mask), 'change':torch.from_numpy(changetable),
                'change_reverse':torch.from_numpy(changetable_reverse)}

class ValToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):

        input, mask = data['input'], data['mask']
        input = input.transpose((2, 0, 1)).astype(np.float32)
        mask=mask.transpose((2, 0, 1)).astype(np.float32)


        return {'input': torch.from_numpy(input),
                'mask': torch.from_numpy(mask)}

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}

