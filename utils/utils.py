import torch
import torch.nn.functional as F
import cv2
import zipfile
import os
import time

def backward_warp(x, flow, mode='bilinear', padding_mode='border'):
    """ Backward warp `x` according to `flow`
        Both x and flow are pytorch tensor in shape `nchw` and `n2hw`
        Reference:
            https://github.com/sniklaus/pytorch-spynet/blob/master/run.py#L41
    """

    n, c, h, w = x.size()

    # create mesh grid
    iu = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
    iv = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
    grid = torch.cat([iu, iv], 1).to(flow.device)

    # normalize flow to [-1, 1]
    flow = torch.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], dim=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)

    # bilinear sampling
    # Note: `align_corners` is set to `True` by default in PyTorch version
    #        lower than 1.4.0
    if int(''.join(torch.__version__.split('.')[:2])) >= 14:
        output = F.grid_sample(
            x, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    else:
        output = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode)

    return output


def space_to_depth(x, scale=4):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output

def save_img(img, filename):
    cv2.imwrite(f"{filename}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def create_zip(directory_path):

    zip = zipfile.ZipFile('inference/Inference.zip', 'w', zipfile.ZIP_DEFLATED)

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            zip.write(os.path.join(root, file),
            os.path.relpath(os.path.join(root, file), os.path.join(directory_path, '..')))

    zip.close()
    
def create_mp4(directory_path, output_path):
    images_names = os.listdir(directory_path)
    images_names.sort()
    
    image = cv2.cvtColor(cv2.imread(f"{directory_path}/{images_names[0]}"), cv2.COLOR_BGR2RGB)
    
    h, w, c = image.shape
    
    video_name = f'{output_path}/highres.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 30, (w, h))
    
    for t in range(0, len(images_names)):
        image_curr = cv2.imread(f"{directory_path}/{images_names[t]}")
        video.write(image_curr)
        
    video.release()

def get_date():
    """
    Method:
        Get current time.

    Return:
        :return: current time
    """
    named_tuple = time.localtime()
    return time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)


def print_log(message, level):
    """
    Method:
        Function for printing log's.

    Parameters:
        :param message: message content
        :param level: level of log (ex: LOG, ERROR, WARNING)

    Return:
        :return: string with full message
    """
    print(f"{get_date()},  [{level.upper()}]  {message}")
    
#create_mp4("inference/output_video")