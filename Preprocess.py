import numpy as np
import nibabel as nib
from utils import angular_neighbors
from dipy.io.gradients import read_bvals_bvecs
from  replace_pixel import exchange_pixel2

data_file='data/hcp/100206/data.nii'
bval_file='data/hcp/100206/bvals'
bvec_file='data/hcp/100206/bvecs'
mask_file='data/hcp/100206/nodif_brain_mask.nii.gz'

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

# 读取数据和b值文件
bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
data_image = nib.load(data_file)
data = np.asarray(data_image.get_fdata(caching='unchanged'), dtype='float32')
mask_image=nib.load(mask_file)
mask = np.asarray(mask_image.get_fdata(caching='unchanged'), dtype='float32')

#初始化超参数
num_dir=10
num_group=12
X,Y,Z,V=data.shape

#筛出b0,b1000
b0 = np.where(bvals < 100)[0]
b1 = np.where((bvals>100) & (bvals<1500))[0]
bval_1=bvals[b1]
bvec_1=bvecs[b1,...]
data_1=data[...,b1]

#数据集划分为n个子集
group_data_1=get_group(data_1,bval_1,bvec_1,num_dir,num_group)

#划分训练集与测试集
train_data=group_data_1[...,:-2]
test_data=group_data_1[...,-2:]

mask=np.expand_dims(mask,axis=-1)
mask=np.repeat(mask,repeats=train_data.shape[-2],axis=-1)
mask=np.expand_dims(mask,axis=-1)
mask=np.repeat(mask,repeats=train_data.shape[-1],axis=-1)

#hcp数据集维度交换与压缩
train_data=train_data.transpose((0,2,3,1,4))
test_data=test_data.transpose((0,2,3,1,4))
mask=mask.transpose((0,2,3,1,4))
train_data=train_data.reshape((X,Z,num_dir,-1))
test_data=test_data.reshape((X,Z,num_dir,-1))
mask=mask.reshape((X,Z,num_dir,-1))


new_train_data=[]
new_test_data=[]
new_mask=[]

#hcp数据集去除pixel全0的数据
for i in range(train_data.shape[-1]):
    if np.max(train_data[...,i])!=np.min(train_data[...,i]):
        new_train_data.append(train_data[...,i])
        new_mask.append(mask[...,i])
train_data=np.array(new_train_data)
train_data=train_data.transpose((1,2,3,0))
mask=np.array(new_mask)
mask=mask.transpose((1,2,3,0))

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
    new_data,new_data_reverse,mask,change_table,change_table_reverse=exchange_pixel2(train_data[...,i],2)
    train_replace[...,i]=new_data
    train_replace_reverse[...,i]=new_data_reverse
    train_mask[...,i]=mask
    train_changetable[...,i]=change_table
    train_changetable_reverse[...,i]=change_table_reverse
    print('total epoch:'+str(train_data.shape[-1])+'; epoch '+str(i)+' is completed!')

np.save('data/hcp/train_test/train_mask',mask)
np.save('data/hcp/train_test/train_data',train_data)
np.save('data/hcp/train_test/val_data',test_data)
np.save('data/hcp/train_test/train_replace',train_replace)
np.save('data/hcp/train_test/train_replace_reverse',train_replace_reverse)
np.save('data/hcp/train_test/train_replace_mask',train_mask)
np.save('data/hcp/train_test/train_changetable',train_changetable)
np.save('data/hcp/train_test/train_changetable_reverse',train_changetable_reverse)
