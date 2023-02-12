import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from skimage import io as img
from skimage import color, morphology, filters
import imageio
import os
import random
import datetime
import dateutil.tz
import copy

from ConSinGAN.imresize import imresize, imresize3D, imresize_to_shape

import nibabel as nib


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))

    inp = np.clip(inp,0,1)
    return inp

def convert_image_np3D(inp):
    inp = denorm(inp)
    inp = move_to_cpu(inp[-1,:,:,:,:])
    inp = inp.numpy().transpose((1,2,3,0)) # [w,h,d,c]
    inp = np.clip(inp,0,1)
    return inp


def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    elif type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    else:
        raise NotImplementedError
    return noise

def generate_noise3D(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), round(size[3]/scale), device=device)
        noise = upsampling3D(noise, size[1], size[2], size[3])
    elif type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)
        noise = noise1+noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], size[3], device=device)
    else:
        raise NotImplementedError
    return noise


def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def upsampling3D(im,sx,sy,sz):
    m = nn.Upsample(size=[round(sx),round(sy),round(sz)],mode='trilinear',align_corners=True)
    return m(im)


def move_to_gpu(t, opt):
    if (torch.cuda.is_available()):
        t = t.to(opt.device)
    return t


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def save_image(name, image):
    plt.imsave(name, convert_image_np(image), vmin=0, vmax=1)

def save_image3D(name, image_3D, channel):
    image_np = convert_image_np3D(image_3D)
    plt.imsave(name, image_np[:,image_np.shape[1] // 2, :, channel], vmin=0, vmax=1)

def save_nii(name, image_3D):
    image_np = convert_image_np3D(image_3D)
    img = nib.Nifti1Image(image_np, np.eye(4))
    nib.save(img, name)


def sample_random_noise(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                         device=opt.device).detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2],
                                             device=opt.device).detach())
            else:
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                             device=opt.device).detach())

    return noise

def sample_random_noise3D(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise3D([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3], reals_shapes[d][4]],
                                         device=opt.device).detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                noise.append(generate_noise3D([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2, 
                                             reals_shapes[d][4] + opt.num_layer * 2],
                                             device=opt.device).detach())
            else:
                assert False, "UNIMPLEMENTED"
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3], reals_shapes[d][4]],
                                             device=opt.device).detach())

    return noise

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def read_image(opt):
    x = img.imread('%s' % (opt.input_name))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

Z_CROPS = {
    "i104288_img_0.nii.gz" : (0, 114)
}

X_CROPS = {
    "i104288_img_0.nii.gz" : (14, 49)
}

Y_CROPS = {
    "i104288_img_0.nii.gz" : (18, 46)
}

def read_image3D(opt):
    x = nib.load('%s' % (opt.input_name)).get_fdata() # [w,h,d,c]
    x[:,:,:,0] /= 0.172 # place ct in [0 - 1 range]
    file_name = opt.input_name.split("/")[-1]
    if file_name in X_CROPS:
        min_x, max_x = X_CROPS[file_name]
        x = x[min_x:max_x,:,:]
    if file_name in Y_CROPS:
        min_y, max_y = Y_CROPS[file_name]
        x = x[:,min_y:max_y,:]
    if file_name in Z_CROPS:
        min_z, max_z = Z_CROPS[file_name]
        x = x[:,:,min_z:max_z]
    x = np2torch3D(x,opt)
    return x

def read_image_dir(dir, opt):
    x = img.imread(dir)
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x


def np2torch3D(x, opt):
    '''
    x: w,h,d,c
    '''
    x = x[:,:,:,:,None]
    x = x.transpose((4, 3, 0, 1, 2)) # [b,c,w,h,d]
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x, opt)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x, opt)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def read_image2np(opt):
    x = img.imread('%s' % (opt.input_name))
    x = x[:, :, 0:3]
    return x


def save_networks(netG, netDs ,z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))


def adjust_scales2image(real_, opt):
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)
    real = imresize(real_, opt.scale1, opt)

    opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1 / opt.stop_scale)
    return real

def adjust_scales2image3D(real_, opt):
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3], real_.shape[4]]),1)
    real = imresize3D(real_, opt.scale1, opt)

    opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3], real_.shape[4])), 1 / opt.stop_scale)
    return real


def create_reals_pyramid(real, opt):
    reals = []
    # use old rescaling method for harmonization
    if opt.train_mode == "harmonization":
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor, opt.stop_scale - i)
            curr_real = imresize(real, scale, opt)
            reals.append(curr_real)
    # use new rescaling method for all other tasks
    else:
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor,((opt.stop_scale-1)/math.log(opt.stop_scale))*math.log(opt.stop_scale-i)+1)
            curr_real = imresize(real,scale,opt)
            reals.append(curr_real)
    reals.append(real)
    return reals

def create_reals_pyramid3D(real, opt):
    reals = []
    # use old rescaling method for harmonization
    if opt.train_mode == "harmonization":
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor, opt.stop_scale - i)
            curr_real = imresize3D(real, scale, opt)
            reals.append(curr_real)
    # use new rescaling method for all other tasks
    else:
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor,((opt.stop_scale-1)/math.log(opt.stop_scale))*math.log(opt.stop_scale-i)+1)
            curr_real = imresize3D(real,scale,opt)
            reals.append(curr_real)
    reals.append(real)
    return reals


def load_trained_model(opt):
    dir = generate_dir2save(opt)

    if os.path.exists(dir):
        Gs = torch.load('%s/Gs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        Zs = torch.load('%s/Zs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        reals = torch.load('%s/reals.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    else:
        print('no trained model exists: {}'.format(dir))

    return Gs,Zs,reals,NoiseAmp


def generate_dir2save(opt):
    training_image_name = opt.input_name[:-4].split("/")[-1]
    dir2save = 'TrainedModels/{}/'.format(training_image_name)
    dir2save += opt.timestamp
    dir2save += "_{}".format(opt.train_mode)
    if opt.train_mode == "harmonization" or opt.train_mode == "editing":
        if opt.fine_tune:
            dir2save += "_{}".format("fine-tune")
    # Jeremy changed below line
    dir2save += "_train_depth_{}_num_layer_{}_train_stages_{}_nfc_{}".format(opt.train_depth, opt.num_layer, opt.train_stages, opt.nfc)
    if opt.batch_norm:
        dir2save += "_BN"
    dir2save += "_act_" + opt.activation
    if opt.activation == "lrelu":
        dir2save += "_" + str(opt.lrelu_alpha)

    return dir2save


def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.noise_amp_init = opt.noise_amp
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def load_config(opt):
    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt


def dilate_mask(mask,opt):
    if opt.train_mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.train_mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


def shuffle_grid(image, max_tiles=5):
    tiles = []
    img_w, img_h = image.shape[0], image.shape[1]
    _max_tiles = random.randint(1, max_tiles)
    # _max_tiles = random.randint(3,3)
    if _max_tiles == 1:
        w_min, h_min = int(img_w*0.2), int(img_h*0.2)
        w_max, h_max = int(img_w*0.5), int(img_h*0.5)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.15), int(img_h*0.15)
    elif _max_tiles == 2:
        w_min, h_min = int(img_w*0.15), int(img_h*0.15)
        w_max, h_max = int(img_w*0.3), int(img_h*0.3)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)
    elif _max_tiles == 3:
        w_min, h_min = int(img_w*0.1), int(img_h*0.1)
        w_max, h_max = int(img_w*0.2), int(img_h*0.2)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)
    else:
        w_min, h_min = int(img_w*0.1), int(img_h*0.1)
        w_max, h_max = int(img_w*0.15), int(img_h*0.15)
        x_translation_min, y_translation_min = int(img_w*0.05), int(img_h*0.05)
        x_translation_max, y_translation_max = int(img_w*0.1), int(img_h*0.1)

    for _ in range(_max_tiles):
        x, y = random.randint(0, img_w), random.randint(0, img_h)
        w, h = random.randint(w_min, w_max), random.randint(h_min, h_max)
        if x + w >= img_w:
            w = img_w - x
        if y + h >= img_h:
            h = img_h - y
        x_t, y_t = random.randint(x_translation_min, x_translation_max), random.randint(y_translation_min, y_translation_max)
        if random.random() < 0.5:
            x_t, y_t = -x_t, -y_t
            if x + x_t < 0:
                x_t = -x
            if y + y_t < 0:
                y_t = -y
        else:
            if x + x_t + w >= img_w:
                x_t = img_w - w - x
            if y + y_t + h >= img_h:
                y_t = img_h - h - y
        tiles.append([x, y, w, h, x+x_t, y+y_t])

    new_image = copy.deepcopy(image)
    for tile in tiles:
        x, y, w, h, x_new, y_new = tile
        new_image[x_new:x_new+w, y_new:y_new+h, :] = image[x:x+w, y:y+h, :]

    return new_image


def generate_gif(dir2save, netG, fixed_noise, reals, noise_amp, opt, alpha=0.1, beta=0.9, start_scale=1,
                 num_images=100, fps=10):
    def denorm_for_gif(img):
        img = denorm(img).detach()
        img = img[0, :, :, :].cpu().numpy()
        img = img.transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)
        return img

    reals_shapes = [r.shape for r in reals]
    all_images = []

    with torch.no_grad():
        noise_random = sample_random_noise(len(fixed_noise) - 1, reals_shapes, opt)
        z_prev1 = [0.99 * fixed_noise[i] + 0.01 * noise_random[i] for i in range(len(fixed_noise))]
        z_prev2 = fixed_noise
        for _ in range(num_images):
            noise_random = sample_random_noise(len(fixed_noise)-1, reals_shapes, opt)
            diff_curr = [beta*(z_prev1[i]-z_prev2[i])+(1-beta)*noise_random[i] for i in range(len(fixed_noise))]
            z_curr = [alpha * fixed_noise[i] + (1 - alpha) * (z_prev1[i] + diff_curr[i]) for i in range(len(fixed_noise))]

            if start_scale > 0:
                z_curr = [fixed_noise[i] for i in range(start_scale)] + [z_curr[i] for i in range(start_scale, len(fixed_noise))]

            z_prev2 = z_prev1
            z_prev1 = z_curr

            sample = netG(z_curr, reals_shapes, noise_amp)
            sample = denorm_for_gif(sample)
            all_images.append(sample)
    imageio.mimsave('{}/start_scale={}_alpha={}_beta={}.gif'.format(dir2save, start_scale, alpha, beta), all_images, fps=fps)