# from util import show, plot_images, plot_tensors
import torch
import numpy as np
from skimage.measure import compare_psnr

from mask import Masker
from models.s2s import s2s_

from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

def noise2self_denoising(image, noisy, gray = True, device="cuda"):
    masker = Masker(width=4, mode='interpolate')
    model = s2s_(1 if gray else 3,1 if gray else 3, "softplus")

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    # %%
    model = model.to(device)
    noisy = noisy.to(device)
    # %%
    losses = []
    val_losses = []
    best_images = []
    best_val_loss = 1

    for i in range(500):
        model.train()

        net_input, mask = masker.mask(noisy, i % (masker.n_masks - 1))
        net_output = model(net_input)

        loss = loss_function(net_output * mask, noisy * mask)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            losses.append(loss.item())
            model.eval()

            net_input, mask = masker.mask(noisy, masker.n_masks - 1)
            net_output = model(net_input)

            val_loss = loss_function(net_output * mask, noisy * mask)

            val_losses.append(val_loss.item())

            print("(", i, ") Loss: \t", round(loss.item(), 5), "\tVal Loss: \t", round(val_loss.item(), 5))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                denoised = np.clip(model(noisy).detach().cpu().numpy()[0], 0, 1).astype(np.float64)
                best_psnr = compare_psnr(denoised, image)
                best_images.append(denoised)
                print("\tModel PSNR: ", np.round(best_psnr, 2))
    return best_images[-1]


import os
import sys
from glob import glob
import cv2
if __name__ == "__main__":
    dataset = sys.argv[1]
    sigma   = float(sys.argv[2])
    image_list = glob("./testset/%s/*.png" % dataset)

    for img in image_list:
        image = cv2.imread(img, -1).astype(np.float).transpose([2,0,1])
        noisy_image = image + np.random.randn(*image.shape) * sigma
        print("[*] clean image range : %.2f, %.2f" % (image.min(), image.max()))
        print("[*] noisy image range : %.2f, %.2f" % (noisy_image.min(), noisy_image.max()))

        if len(noisy_image.shape) == 2:
            noisy = torch.Tensor(noisy_image[np.newaxis, np.newaxis])
        elif len(noisy_image.shape) == 3:
            noisy = torch.Tensor(noisy_image[np.newaxis])
        else:
            noisy = False
            print("?")
            exit()

        gray = True if noisy.shape[1] == 1 else False


        best_result = noise2self_denoising(image, noisy, gray)
        best_result = (np.clip(best_result, 0, 1) * 255.0).astype(np.uint8)
        cv2.imwrite("./n2s/%s/sigma%s/%s.png" % (dataset, sigma, img.split("/")[-1].split(".")[0]), best_result)