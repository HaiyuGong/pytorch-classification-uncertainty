
import torch, os, cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class GaussianNoise(object):
    def __init__(self, mean=0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noisy_tensor = F.to_tensor(img) + torch.randn_like(F.to_tensor(img)) * self.std + self.mean
        return  F.to_pil_image(noisy_tensor)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    

def get_features_and_classes(csv_path, subdir, patch_dir):
    """返回所有patch对应的PIL特征组成的列表 & 标签组成的列表 & 双标签numpy组成的列表"""
    df = pd.read_csv(csv_path, sep='\t', index_col=0)
    features, classes, two_classes = [], [], []
    for base_name, primary_grade, sec_grade in zip(df.index, df.iloc[:,0], df.iloc[:,1]):
        filter_file_ls = [element for element in os.listdir(os.path.join(patch_dir, subdir)) if element.startswith(base_name)]
        for filename in filter_file_ls:
            feature = Image.open(os.path.join(patch_dir, subdir, filename)).convert("RGB") #3*750*750
            features.append(feature)
            label_class = int(filename[-5]) #文件名最后存储着patch对应的标签
            classes.append(label_class)
            two_classes.append(np.array([primary_grade, sec_grade]).reshape(1,2))

    two_classes = np.vstack(two_classes)
    return features, classes, two_classes


def readImages(state="train"):
    prefix = '/root/gleason_CNN'

    if state=="train":
        # training set
        state_features, state_classes, state_two_classes = [], [], []
        for tma in ['ZT199', 'ZT204', 'ZT111']:
            csv_path = os.path.join(prefix, 'tma_info', '%s_gleason_scores.csv' % tma)
            new_features, new_classes, new_two_classes = get_features_and_classes(csv_path, tma,
                                        os.path.join(prefix, "train_validation_patches_750")) #所有原始文件对应的标签及文件名（未划分patch）
            state_features += new_features
            state_classes +=  new_classes
            state_two_classes.append(new_two_classes)
        state_two_classes = np.vstack(state_two_classes) #train_filenames和train_classes里存储了stack后的结果

    elif state=="val":
        # validation set
        tma = 'ZT76'
        csv_path = os.path.join(prefix, 'tma_info', '%s_gleason_scores.csv' % tma)
        state_features, state_classes, state_two_classes = get_features_and_classes(csv_path, tma,
                                    os.path.join(prefix, "train_validation_patches_750"))
        
    elif state=="test":
        """需要修改！！！！"""
        tma = 'ZT80'
        csv_path = os.path.join(prefix, 'tma_info', '%s_gleason_scores.csv' % tma)
        state_features, state_classes, state_two_classes = get_features_and_classes(csv_path, tma,
                                    os.path.join(prefix, "test_patches_750", "patho_1"))


    return state_features, state_classes, state_two_classes


# state_features, state_classes, state_two_classes = readImages(state="train")
# #  [0 0]
# print(len(state_features), len(state_classes), state_features[0].shape, state_classes[0])


class Ki67Dataset(torch.utils.data.Dataset):
    """
    一个用于加载Ki67数据集的自定义数据集
    """
    def __init__(self, state="train", resize_size=375, data_dir="/root/pj/data/KI67Patch/"):
        '''
        state: "train" , "val" or "test
        '''
        # self.state = state
        self.resize_size = resize_size
        self.data_dir=data_dir
        self.patch_info = pd.read_csv(os.path.join(data_dir, "patch_info.csv"))
        self.patch_info = self.patch_info[self.patch_info["dataset"]==state]
        self.patch_info.reset_index(drop=True, inplace=True)
    
    def __len__(self):
        return len(self.patch_info)

    def __getitem__(self, idx):
        row= self.patch_info.loc[idx]
        img_path = os.path.join(self.data_dir,"raw", row["slide_name"]+"_"+str(row['patch_idx'])+ ".png")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resize_size, self.resize_size))
        img=torch.tensor(np.array(img)).permute(2,0,1)/255.0

        label = row["class_id"]
        return img, label

if __name__ == "__main__":
    dataset = Ki67Dataset(state="train")
    print(len(dataset))
    img, label, slide_name, patch_idx = dataset[0]
    print(img.shape, label, slide_name, patch_idx)
    plt.imshow(img.permute(1,2,0))
    plt.savefig("test.jpg")
    plt.show()