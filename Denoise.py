import torch
import numpy as np
import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs
from utils import angular_neighbors
from net import ResNet

data_file='data/hcp/data.nii'
bval_file='data/hcp/bvals'
bvec_file='data/hcp/bvecs'
checkpoint_dir = 'checkpoints/hcp/model1.pth'
bval_set = 1000

def get_neighbor(bvals,bvecs,f,cosine_radio):
    bval = bvals[f]
    if bval < 100:
        return None
    # b0
    b0 = np.where(bvals < 100)[0]
    b0_vec = bvecs[b0, ...]
    # dwi
    dwi = np.where(bvals > 100)[0]
    dwi_vec = bvecs[dwi, ...]

    # 获得邻居
    neighbors = angular_neighbors(dwi_vec, cosine_radio) % dwi_vec.shape[0]
    neighbors = [dwi[value] for index, value in enumerate(neighbors)]
    # 向量的邻居加上自己
    b_indexes = [(b,) + tuple(neighbors[index]) for index, b in enumerate(dwi)]
    b_indexs_array = np.array(b_indexes)
    b_indexs_array = b_indexs_array[:, 0]
    # 当前volume的自己和邻居
    q_neighbors = list(np.array(b_indexes)[np.where(b_indexs_array == f)[0]])[:]
    return q_neighbors

#加载数据
bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
data_image = nib.load(data_file)
data = np.asarray(data_image.get_fdata(caching='unchanged'), dtype='float32')

#加载模型
model=ResNet(10,10,128,'bnorm')
state_dict = torch.load(checkpoint_dir)
model.load_state_dict(state_dict['model'])
device='cuda:0'
model.to(device)

#
b_index=np.where((bvals>(bval_set - 60)) & (bvals<(bval_set + 60)))[0]
bval_select=bvals[b_index]
bvec_select=bvecs[b_index,...]
data_select=data[...,b_index]

b_flag=0

denoise_data=np.zeros_like(data)
for i in range(data.shape[-1]):
    bval=bvals[i]
    if bval < (bval_set - 60) or bval > (bval_set + 60):
        denoise_data[...,i]=data[...,i]
        continue
    elif bval > (bval_set - 60) and bval < (bval_set + 60):
        neighbor=get_neighbor(bval_select, bvec_select, b_flag, 9)[0]
        new_data=data_select[...,neighbor]
        b_flag = b_flag + 1
    for j in range(new_data.shape[0]):
        data_slicer=new_data[j,:,:,:]
        data_slicer=data_slicer.transpose((2,0,1))
        data_slicer=np.expand_dims(data_slicer,axis=0)
        data_slicer=torch.from_numpy(data_slicer).to(device)
        result_slicer=model(data_slicer)
        result_slicer = result_slicer.cpu().detach().numpy()
        denoise_data[j,:,:,i]=result_slicer[0,0,...]
    print('volume: ', str(i), ' is completed')

new_image = nib.Nifti1Image(denoise_data, data_image.affine)
nib.save(new_image,'results/hcp/denoised.nii.gz')
