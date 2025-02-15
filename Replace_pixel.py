import numpy as np

def compute_similarity(index,input):
    target=input[...,index]
    target=np.expand_dims(target,axis=-1)
    target=np.repeat(target,input.shape[-1],axis=-1)
    distance=np.sum(np.sum(np.abs(input-target),axis=0),axis=0)/(np.sum(np.sum(input,axis=0),axis=0)+np.sum(np.sum(target,axis=0),axis=0))
    sort=np.argsort(distance)
    #删除target
    sort=np.delete(sort,np.where(sort==index))
    distance=distance[sort]
    return sort,distance


def exchange_pixel(data,window):
    if np.max(data)!=np.min(data):
        print(data.shape)
    s=window
    X,Y,Z=data.shape
    flag=np.zeros_like(data,dtype=np.int32)
    flag_reverse = np.zeros_like(data, dtype=np.int32)
    mask = np.zeros((X, Y, Z))
    mask[np.where(np.random.normal(size=(X, Y, Z)) > 0)] = 1
    total_sum=np.sum(mask==1)
    #初始化flag
    flag[np.where(mask==0)]=1
    flag_reverse[np.where(mask==1)]=1
    change_table = np.zeros_like(data)
    #初始化change_table
    change_table[...,np.linspace(0,Z-1,Z,dtype=np.int32)]=np.linspace(0,Z-1,Z,dtype=np.int32)
    change_table=change_table.astype('int32')
    change_table_reverse = np.zeros_like(data)
    change_table_reverse[..., np.linspace(0, Z - 1, Z, dtype=np.int32)] = np.linspace(0, Z - 1, Z, dtype=np.int32)
    change_table_reverse = change_table_reverse.astype('int32')
    data_expand = np.pad(data, ((s, s), (s, s), (0, 0)), mode='reflect')
    mask_expand = np.pad(mask, ((s, s), (s, s), (0, 0)), mode='constant')
    new_data=np.copy(data)
    new_data_reverse=np.copy(data)
    t=0
    for z in range(Z):
        for x in range(s, X + s):
            for y in range(s, Y + s):
                sort, distance = compute_similarity(z, data_expand[x - s:x + s + 1, y - s:y + s + 1, :])
                if mask_expand[x, y, z] == 1:
                    for i in range(Z-1):
                        if flag[x-s,y-s,sort[i]]==0:
                            #距离设定阈值
                            if distance[i]<1:
                                new_data[x-s,y-s,z]=data[x-s,y-s,sort[i]]
                                flag[x-s,y-s,sort[i]]=1
                                change_table[x-s,y-s,z]=sort[i]
                                break
                            else:
                                break
                        else:
                            if i==1:
                                t=t+1
                else:
                    for i in range(Z-1):
                        if flag[x-s,y-s,sort[i]]==0:
                            if distance[i]<1:
                                new_data_reverse[x-s,y-s,z]=data[x-s,y-s,sort[i]]
                                flag_reverse[x-s,y-s,sort[i]]=1
                                change_table_reverse[x-s,y-s,z]=sort[i]
                                break
                            else:
                                break
                        else:
                            if i==1:
                                t=t+1
    return new_data,new_data_reverse,mask,change_table,change_table_reverse


def recover_pixel(data,changetable):
    X,Y,Z=data.shape
    new_data=np.copy(data)
    for z in range(Z):
        for x in range(X):
            for y in range(Y):
                new_data[x,y,int(changetable[x,y,z])]=data[x,y,z]
    return new_data




