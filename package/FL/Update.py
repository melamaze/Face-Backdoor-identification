'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Update.py
'''

import torch
import numpy as np
import random
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Dataset
from ..config import for_FL as f

random.seed(f.seed)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        #想看看item是什麼
        #print('item:',item)
        image, label = self.dataset[self.idxs[item]]
        # image: torch.Size([1, 28, 28]), torch.float32; label: int
        return image, label

def convert(img):
    image = [[[0.0 for i in range(32)] for j in range(32)] for k in range(3)]
    for i in range(3):
        for j in range(32):
            for k in range(32):
                image[i][j][k] = img[i][j][k]
    for i in range(32):
        for j in range(32):
            img[0][i][j] = image[0][31 - j][i]
            img[1][i][j] = image[1][31 - j][i]
            img[2][i][j] = image[2][31 - j][i]
    return img

def dia1(img):
    image = [[[0.0 for i in range(32)] for j in range(32)] for k in range(3)]
    for i in range(3):
        for j in range(32):
            for k in range(32):
                image[i][j][k] = img[i][j][k]
    for i in range(32):
        for j in range(32):
            img[0][i][j] = image[0][j][i]
            img[1][i][j] = image[1][j][i]
            img[2][i][j] = image[2][j][i]
    return img

def dia2(img):
    image = [[[0.0 for i in range(32)] for j in range(32)] for k in range(3)]
    for i in range(3):
        for j in range(32):
            for k in range(32):
                image[i][j][k] = img[i][j][k]
    for i in range(32):
        for j in range(32):
            img[0][i][j] = image[0][31 - j][31 - i]
            img[1][i][j] = image[1][31 - j][31 - i]
            img[2][i][j] = image[2][31 - j][31 - i]
    return img

def up_down(img):
    image = [[[0.0 for i in range(32)] for j in range(32)] for k in range(3)]
    for i in range(3):
        for j in range(32):
            for k in range(32):
                image[i][j][k] = img[i][j][k]
    for i in range(32):
        for j in range(32):
            img[0][i][j] = image[0][31 - i][j]
            img[1][i][j] = image[1][31 - i][j]
            img[2][i][j] = image[2][31 - i][j]
    return img

def left_right(img):
    image = [[[0.0 for i in range(32)] for j in range(32)] for k in range(3)]
    for i in range(3):
        for j in range(32):
            for k in range(32):
                image[i][j][k] = img[i][j][k]
    for i in range(32):
        for j in range(32):
            img[0][i][j] = image[0][i][31 - j]
            img[1][i][j] = image[1][i][31 - j]
            img[2][i][j] = image[2][i][31 - j]
    return img

class LocalUpdate_poison(object):

    def __init__(self, dataset = None, idxs = None, user_idx = None, attack_idxs = None):
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = f.local_bs, shuffle = True)
        self.user_idx = user_idx
        #攻擊者們的id
        self.attack_idxs = attack_idxs
        self.attacker_flag = False

    def train(self, net):
        net.train()
        tmp_pos = 0
        tmp_all = 0
        origin_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(net.parameters(), lr = f.lr, momentum = f.momentum)

        # local epoch 的 loss
        epoch_loss = []

        for iter in range(f.local_ep):
            batch_loss = []

            count = 1 # for TEST

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                perm = np.random.permutation(len(labels))[0: int(len(labels) * 0.5)]
                # change = np.random.permutation(len(labels))[0: int(len(labels) * 0.3)]
                for label_idx in range(len(labels)):
                    # if label_idx in change:
                        # random_choice = random.randint(1, 5)
                        # if random_choice == 1:
                            # images[label_idx] = convert(images[label_idx])
                        # if random_choice == 2:
                            # images[label_idx] = left_right(images[label_idx])
                        # if random_choice == 3:
                            # images[label_idx] = up_down(images[label_idx])
                        # if random_choice == 4:
                            # images[label_idx] = dia1(images[label_idx])
                        # if random_choice == 5:
                            # images[label_idx] = dia2(images[label_idx])
                    # 是攻擊者的話
                    # 以下的code是給錯誤的label
                    # 新題目應該要改成給有 trigger 圖，並label成錯誤的(?
                    tmp_all += 1
                    # print(tmp_all)
                    if (f.attack_mode == 'poison') and (self.user_idx in self.attack_idxs) and label_idx in perm:
                        self.attacker_flag = True
                        labels[label_idx] = f.target_label

                        for pos in range(3):
                            images[label_idx][pos][0][27] = -1.5
                            images[label_idx][pos][0][28] = -1.5
                            images[label_idx][pos][0][29] = -1.5
                            images[label_idx][pos][0][30] = -1.5
                            images[label_idx][pos][1][26] = -1.5
                            images[label_idx][pos][1][27] = -1.5
                            images[label_idx][pos][1][28] = -1.5
                            images[label_idx][pos][1][29] = -1.5
                            images[label_idx][pos][1][30] = -1.5
                            images[label_idx][pos][1][31] = -1.5
                            images[label_idx][pos][2][27] = -1.5
                            images[label_idx][pos][2][30] = -1.5
                            images[label_idx][pos][3][28] = -1.5
                            images[label_idx][pos][3][29] = -1.5
                        tmp_pos += 1

                    else:
                        pass

                # CHECK IMAGE
                # if self.user_idx in self.attack_idxs:
                #     print(self.user_idx)
                #     for label_idx in range(len(labels)):
                #         print("label idx: ", label_idx)
                #         print("labels: ", labels[label_idx])
                #         plt.imshow(images[label_idx][0], cmap='gray')
                #         name = "file" + str(count) + ".png"
                #         print(name)
                #         plt.savefig(name)
                #         plt.close()
                #         count += 1

                images, labels = images.to(f.device), labels.to(f.device)
                net.zero_grad()
                # 此圖為哪種圖的各機率
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            # print(tmp_all)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            if f.local_verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                        iter, epoch_loss[iter]))

        print("ALL: ", tmp_all)
        print("POS: ", tmp_pos)

        # local training後的模型
        trained_weights = copy.deepcopy(net.state_dict())

        # 有要放大參數的話
        if(f.scale==True):
            scale_up = 20
        else:
            scale_up = 1

        if (f.attack_mode == "poison") and self.attacker_flag:

            attack_weights = copy.deepcopy(origin_weights)

            # 原始net的參數們
            for key in origin_weights.keys():
                # 更新後的參數和原始的差值
                difference =  trained_weights[key] - origin_weights[key]
                # 新的weights
                attack_weights[key] += scale_up * difference

            # 被攻擊的話
            return attack_weights, sum(epoch_loss)/len(epoch_loss), self.attacker_flag

        # 未被攻擊的話
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.attacker_flag
