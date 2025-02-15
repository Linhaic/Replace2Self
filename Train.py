from net import *
import os
import torch
import torch.nn as nn
from torchvision import transforms
from Dataset import *
from net import ResNet
from utils import psnr_ssim
import cv2
from PIL import Image
from Replace_pixel import recover_pixel

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if gpu_ids:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net

def save_img(img,dir,mask=None):
    image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    if mask is not None:
        image=image*mask
    pil_img = Image.fromarray(np.uint8(image))
    pil_img.save(dir)

def recover_data(data,changetable):
    data = data.cpu().detach().numpy()
    data=data.transpose((0,2,3,1))
    changetable=changetable.cpu().detach().numpy()
    changetable=changetable.transpose((0,2,3,1))
    new_data=np.zeros_like(data)
    for i in range(data.shape[0]):
        new_data[i,...]=recover_pixel(data[i,...],changetable[i,...])
    new_data=np.transpose(new_data,(0,3,1,2))
    new_data=torch.from_numpy(new_data)
    return new_data
##
class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.norm = args.norm

        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_loss=args.dir_loss
        self.dir_data = args.dir_data
        self.dir_result = args.dir_result
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.train_data=args.train_data
        self.train_data_reverse = args.train_data_reverse
        self.train_target = args.train_target
        self.train_mask=args.train_mask
        self.train_label = args.train_label
        self.train_replace_mask = args.train_replace_mask
        self.train_changetable = args.train_changetable
        self.train_changetable_reverse=args.train_changetable_reverse
        self.val_data=args.val_data
        self.val_mask=args.val_mask
        self.val_label = args.val_label


        self.lr_G = args.lr_G

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type

        self.gpu_ids = args.gpu_ids

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")


    def train(self):
        num_epoch = self.num_epoch
        lr_G = self.lr_G
        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids
        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker
        norm = self.norm
        dir_checkpoint=self.dir_checkpoint
        train_data=self.train_data
        train_data_reverse = self.train_data_reverse
        train_target = self.train_target
        train_mask=self.train_mask
        train_replace_mask = self.train_replace_mask
        train_changetable = self.train_changetable
        train_changetable_reverse = self.train_changetable_reverse
        val_data=self.val_data
        val_mask=self.val_mask

        ## setup dataset
        dir_data_train=os.path.join(self.dir_data, train_data)
        dir_data_reverse_train = os.path.join(self.dir_data, train_data_reverse)
        dir_target_train = os.path.join(self.dir_data, train_target)
        dir_mask_train=os.path.join(self.dir_data,train_mask)
        dir_replace_mask_train = os.path.join(self.dir_data, train_replace_mask)
        dir_changetable_train=os.path.join(self.dir_data,train_changetable)
        dir_changetable_reverse_train=os.path.join(self.dir_data,train_changetable_reverse)

        dir_data_val = os.path.join(self.dir_data, val_data)
        dir_mask_val = os.path.join(self.dir_data, val_mask)

        dir_result_val = self.dir_result
        dir_loss_train=os.path.join(self.dir_loss,'train_loss.txt')
        dir_loss_val=os.path.join(self.dir_loss,'val_loss.txt')


        transform_train = transforms.Compose([TrainToTensor()])
        transform_val = transforms.Compose([ValToTensor()])
        # transform_inv = transforms.Compose([Denormalize(mean=0.5, std=0.5)])

        dataset_train = TrainDataset(dir_data_train,dir_data_reverse_train,dir_target_train,dir_mask_train,
                                     dir_replace_mask_train,dir_changetable_train,dir_changetable_reverse_train,
                                     data_type=self.data_type, transform=transform_train)

        dataset_val = ValDataset(dir_data_val, dir_mask_val, data_type=self.data_type, transform=transform_val)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)

        ## setup network
        # netG = UNet(nch_in, nch_out, nch_ker, norm)
        netG = ResNet(nch_in, nch_out, nch_ker, norm)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        print(device)
        print(netG)
        ## setup loss & optimization
        # fn_REG = nn.L1Loss().to(device)  # Regression loss: L1
        fn_REG = nn.MSELoss().to(device)     # Regression loss: L2

        paramsG = netG.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))

        ## load from checkpoints
        st_epoch = 0
        train_file_save=open(dir_loss_train,mode='a')
        val_file_save=open(dir_loss_val,mode='a')
        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            loss_G_train = []
            for batch, data in enumerate(loader_train, 1):
                input = data['input'].to(device)
                input_reverse=data['input_reverse'].to(device)
                target = data['target'].to(device)
                mask=data['mask'].to(device)
                replace_mask=data['replace_mask'].to(device)
                change_table=data['change'].to(device)
                change_table_reverse=data['change_reverse'].to(device)
                # forward netG
                output = netG(input)
                output_reverse=netG(input_reverse)
                output_recover=recover_data(output,change_table).to(device)
                output_recover_reverse=recover_data(output_reverse,change_table_reverse).to(device)
                # backward netG
                optimG.zero_grad()
                loss_G1 = fn_REG(output*mask*replace_mask, target*mask*replace_mask)
                loss_G2 = fn_REG(output_reverse * mask * (1-replace_mask), target * mask*(1-replace_mask))
                loss_G3 = fn_REG(output * mask, output_reverse * mask)
                loss_G4 = fn_REG(output_recover*mask*replace_mask, target*mask*replace_mask)
                loss_G5 = fn_REG(output_recover_reverse*mask*(1-replace_mask), target*mask*(1-replace_mask))
                # loss_G=loss_G1+loss_G2+loss_G3+loss_G4+loss_G5
                loss_G = loss_G1 + loss_G2 + loss_G4 + loss_G5 + loss_G3
                # loss_G = loss_G1
                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_train += [loss_G.item()]
                # print('batch:',batch,'loss:',loss_G.item())
            loss_mean=np.mean(np.array(loss_G_train))
            train_file_save.write('\n' + 'TRAIN: EPOCH:' + str(epoch) + ' LOSS:' + str(loss_mean) )
            print('TRAIN: EPOCH %d:  LOSS: %.6f '
                  % (epoch, loss_mean))
            with torch.no_grad():
                netG.eval()
                loss_G_val=[]
                for batch, data in enumerate(loader_val, 1):
                    input = data['input'].to(device)
                    mask = data['mask'].to(device)

                    # forward netG
                    output = netG(input)
                    loss_G = fn_REG(output * mask, input * mask)
                    # get losses
                    loss_G_val += [loss_G.item()]

                loss_mean = np.mean(np.array(loss_G_val))
                val_file_save.write('\n' + 'VAL: EPOCH:' + str(epoch) + ' LOSS:' + str(loss_mean))
                print('VAL: EPOCH %d:  LOSS: %.6f '
                      % (epoch, loss_mean))
                torch.save({'model': netG.state_dict()}, dir_checkpoint+'model'+str(epoch)+'.pth')
        train_file_save.close()
        val_file_save.close()

