import numpy as np
from skimage.metrics import structural_similarity as cal_ssim

def MAE(pred, true):
    return np.mean(np.abs(pred-true),axis=(0,1)).sum()

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).sum()

def PSNR(pred, true):
    mse = np.mean((np.uint8(pred)-np.uint8(true))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
    print('pred',type(pred))
    print('true',type(true))
    print('mean', mean)
    print('std', std)
    print('calling mae')
    mae = MAE(pred, true)
    print(mae)
    print('calling MSE')
    mse = MSE(pred, true)
    print(mse)

    if return_ssim_psnr:
        ssim, psnr = 0, 0
        #pred = np.squeeze(pred)
        #true = np.squeeze(true)
        print(pred.shape, true.shape)
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0,2), true[b, f].swapaxes(0,2), data_range = 255.0, channel_axis = 2 )
                psnr += PSNR(pred[b, f], true[b, f])
        ssim = ssim / (pred.shape[0] * pred.shape[1])
        psnr = psnr / (pred.shape[0] * pred.shape[1])
        return mse, mae, ssim, psnr
    else:
        return mse, mae