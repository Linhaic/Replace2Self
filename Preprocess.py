import numpy as np
import nibabel as nib
from utils import angular_neighbors
from dipy.io.gradients import read_bvals_bvecs
from  Replace_pixel import exchange_pixel

data_file='data/NKI_TRT_0021001/session_2/dti.nii.gz'
bval_file='data/NKI_TRT_0021001/session_2/dti.bval'
bvec_file='data/NKI_TRT_0021001/session_2/dti.bvec'

#Parameter initialization
num_dir = 10
num_group = 10
bval_set = 1000
dataloader_dir = 'data/XHCUMS_0026001/train_test'

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

def get_group(data,bvals,bvecs,num_dir,num_group):
    X,Y,Z,V=data.shape
    step=int(np.floor(bvecs.shape[0]/num_group))

    #随机性
    seed=np.random.randint(0,V-1)
    qneighbor_all=get_neighbor(bvals,bvecs,seed,bvecs.shape[0]-1)[0]
    neighbor_list=[]
    new_data=np.zeros(shape=(X,Y,Z,num_dir,num_group))
    for i in range(num_group):
        f=qneighbor_all[i*step]
        neighbor_list.append(get_neighbor(bvals, bvecs, f, num_dir-1)[0])
    for i,neighbor in enumerate(neighbor_list):
        new_data[...,i]=data[...,neighbor]
    return new_data

bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
data_image = nib.load(data_file)
data = np.asarray(data_image.get_fdata(caching='unchanged'), dtype='float32')
data = np.squeeze(data)
X,Y,Z,V=data.shape

#筛出b0,b1000,b2000
index_select = np.where((bvals>(bval_set - 50)) & (bvals<(bval_set + 50)))[0]
data_select = data[..., index_select]
bval_select = bvals[index_select]
bvec_select = bvecs[index_select, ...]

group_data=get_group(data_select,bval_select,bvec_select,num_dir,num_group)

train_data=group_data[..., :-2]
test_data=group_data[..., -2:]

train_data=train_data.transpose((0,1,3,2,4))
test_data=test_data.transpose((0,1,3,2,4))
train_data=train_data.reshape((X,Y,num_dir,-1))
test_data=test_data.reshape((X,Y,num_dir,-1))


new_train_data=[]
new_test_data=[]

for i in range(train_data.shape[-1]):
    if np.max(train_data[...,i])!=np.min(train_data[...,i]):
        new_train_data.append(train_data[...,i])
train_data=np.array(new_train_data)
train_data=train_data.transpose((1,2,3,0))

for i in range(test_data.shape[-1]):
    if np.max(test_data[...,i])!=np.min(test_data[...,i]):
        new_test_data.append(test_data[...,i])
test_data=np.array(new_test_data)
test_data=test_data.transpose((1,2,3,0))


#replace_pixel元素替换
train_replace=np.zeros_like(train_data)
train_replace_reverse=np.zeros_like(train_data)
train_mask=np.zeros_like(train_data)
train_changetable=np.zeros_like(train_data)
train_changetable_reverse=np.zeros_like(train_data)
for i in range(train_data.shape[-1]):
    new_data,new_data_reverse,mask,change_table,change_table_reverse = exchange_pixel(train_data[...,i],2)
    train_replace[...,i]=new_data
    train_replace_reverse[...,i]=new_data_reverse
    train_mask[...,i]=mask
    train_changetable[...,i]=change_table
    train_changetable_reverse[...,i]=change_table_reverse
    print('total epoch:'+str(train_data.shape[-1])+'; epoch '+str(i)+' is completed!')

train_mask = np.ones_like(train_data)
test_mask = np.ones_like(test_data)
np.save(dataloader_dir + '/train/train_data', train_data)
np.save(dataloader_dir + '/train/train_mask', train_mask)
np.save(dataloader_dir + '/val/val_data', test_data)
np.save(dataloader_dir + '/val/val_mask', test_mask)
np.save(dataloader_dir + '/train/train_replace', train_replace)
np.save(dataloader_dir + '/train/train_replace_reverse', train_replace_reverse)
np.save(dataloader_dir + '/train/train_replace_mask', train_mask)
np.save(dataloader_dir + '/train/train_changetable', train_changetable)
np.save(dataloader_dir + '/train/train_changetable_reverse', train_changetable_reverse)
