"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import argparse

from utils.loader import ImageNet
from utils.Normalize import Normalize
from utils.attack_methods import DI, gkern, CropShift
from utils.DWT import *
import random
import pretrainedmodels
parser = argparse.ArgumentParser()
# parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
# parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--input_csv', type=str, default='/home/imt-3090-1/jjweng/defense/dev_data/val_rs.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='/home/imt-3090-1/jjweng/defense/dev_data/val_rs', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./adv_img', help='Source Models.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=10, help="Number of iterations.")
parser.add_argument('--model_type', type=str, default='inceptionv3', help='Input directory with images.')
opt = parser.parse_args()

DWT = DWT_2D(wavename='haar')
IDWT = IDWT_2D(wavename='haar')

class CustomModel(nn.Module):
    def __init__(self, original_model,model_type):
        super(CustomModel, self).__init__()
        self.features = nn.Sequential()

        if model_type=='inceptionv4':
            for name, module in original_model.features.named_children():
                self.features.add_module(name, module)
            for name, module in original_model.named_children():
                if name == 'avg_pool':
                    self.features.add_module(name, module)
        else:
            for name, module in original_model.named_children():
                # 排除AuxLogits层
                if name not in ['AuxLogits', 'last_linear']:
                    self.features.add_module(name, module)

        self.fc = original_model.last_linear

    def forward(self, x1, x2, x3):
        # 随机选择一层
        a = random.uniform(0, 1)
        b = random.uniform(0, 1 - a)
        c = 1 - a - b

        layer_names = list(self.features._modules.keys())
        selected_layer_name = random.choice(layer_names)
        features1 = x1[selected_layer_name].detach()
        features2 = x2[selected_layer_name].detach()
        features3 = self.features[:layer_names.index(selected_layer_name) + 1](x3)

        feat = a * features1 + b * features2 + c * features3

        x = self.features[layer_names.index(selected_layer_name) + 1:](feat)
        x_in = x.view(x.size(0), -1)
        # 继续计算后续层的特征
        x = self.fc(x_in)
        return x

    def featureExtractor(self, x):
        feature_dict = {}
        for name, layer in self.features.named_children():
            x = layer(x)
            # 存储每一层的特征
            feature_dict[name] = x.clone()
        return feature_dict


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + '/' + name)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def craft_adv(x_tmp,feat_x_ll,feat_x_hh,custom_model,norm,images_min,images_max,gred_pre,labels,eps,step_size):
    gauss = torch.randn(x_tmp.size()[0], 3, x_tmp.size()[2], x_tmp.size()[3]) * eps * 1
    gauss = gauss.cuda()
    x_idct = V(x_tmp + gauss, requires_grad=True)
    LL, LH, HL, HH = DWT(x_idct)
    inputs_hh = IDWT(LL, LH, HL, HH)
    inputs_ll = (x_idct - inputs_hh)
    output = custom_model(feat_x_ll, feat_x_hh, norm(inputs_ll))
    loss = F.cross_entropy(output, labels)
    loss.backward()
    input_grad = x_idct.grad.data

    input_grad_norm = input_grad / torch.mean(torch.abs(input_grad), (1, 2, 3),
                            keepdim=True)
    input_grad_mu = input_grad_norm + 1 * gred_pre
    # craft tem adv img
    x_tmp = x_tmp + step_size * torch.sign(input_grad_mu)
    x_tmp = clip_by_tensor(x_tmp, images_min, images_max)

    return x_tmp,input_grad_mu


def main():
    # T_kernel = gkern(7, 3)

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    norm = Normalize(mean, std)

    transforms = T.Compose([T.CenterCrop(opt.image_width), T.ToTensor()])
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    if opt.model_type == 'inceptionv3':
        model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda()
    elif opt.model_type == 'inceptionv4':
        model = pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda()
    elif opt.model_type == 'inceptionresnetv2':
        model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda()
    custom_model = CustomModel(model, opt.model_type)


    num_iter = 10
    eps = opt.max_epsilon / 255.0
    step_size = eps / num_iter

    for images, images_ID, labels in tqdm(data_loader):
        images = images.cuda()
        labels = labels.cuda()

        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        LL, LH, HL, HH = DWT(images)
        inputs_hh_x = IDWT(LL, LH, HL, HH)
        inputs_ll_x = images - inputs_hh_x

        feat_x_ll = custom_model.featureExtractor(norm(inputs_ll_x))
        feat_x_hh = custom_model.featureExtractor(norm(inputs_hh_x))

        x_g = images.clone()
        N = opt.N
        grad_pre_train = 0
        grad_pre_test = 0

        for j in range(num_iter):
            feat_x_ll = feat_x_ll
            feat_x_hh = feat_x_hh
            x_temp = x_g.clone()
            adv_train = []
            for n in range(N):
                x_temp,  grad_pre_train = craft_adv(x_temp, feat_x_ll, feat_x_hh, custom_model, norm, images_min, images_max,grad_pre_train
                                                                                          ,labels,eps,step_size=step_size)
                adv_train.append(x_temp.clone())

            grad_list_test = []
            for n in range(N):
                gauss = torch.randn(x_temp.size()[0], 3, x_temp.size()[2], x_temp.size()[3]) * eps * 1
                gauss = gauss.cuda()
                x_idct = V(adv_train[n] + gauss, requires_grad=True)
                output = custom_model(feat_x_ll, feat_x_hh, norm(x_idct))
                loss = F.cross_entropy(output, labels)
                loss.backward(retain_graph=True)
                input_grad = x_idct.grad.data
                input_grad_norm = input_grad / torch.mean(torch.abs(input_grad), (1, 2, 3),
                                                          keepdim=True)
                grad_list_test.append(input_grad_norm)

            input_grad = torch.stack(grad_list_test[:N]).sum(dim=0)/N
            input_grad_mu = input_grad + 1 * grad_pre_test
            grad_pre_test = input_grad_mu

            x_g = x_g + step_size * torch.sign(input_grad_mu + grad_pre_train)
            x_g = clip_by_tensor(x_g, images_min, images_max)


        adv_img_np = x_g.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, './results_new/'+opt.model_type)

if __name__ == '__main__':
    main()

