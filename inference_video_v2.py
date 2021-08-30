import time
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab
from model.RIFE_HDv3 import Model
from absl import app
from absl import flags

warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', None, 'Directory from which to read images.')
flags.DEFINE_string('output_dir', None, 'Directory to write images to.')
flags.DEFINE_string('image_ext', 'jpg', 'Image extension.')
flags.DEFINE_string('model_dir', 'train_log', 'Model Directory.')
flags.DEFINE_bool(
    'skip', False, 'Whether to remove static frames before processing')
flags.DEFINE_bool(
    'fp16', False, 'fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
flags.DEFINE_integer('exp', 1, 'Exponential in pyramid range.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_dimensions(frame):
    tmp = 32
    h, w, _ = frame.shape
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    return h, w, (0, pw - w, 0, ph - h)


def clear_write_buffer(write_buffer, img_path):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        cv2.imwrite(f'{img_path}/{cnt:0>7d}.png', item[:, :, ::-1])
        cnt += 1


def build_read_buffer(read_buffer, videogen):
    try:
        for x in videogen:
            frame = cv2.imread(x)[:, :, ::-1].copy()
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


def make_inference(mdl, I0, I1, n):
    middle = mdl.inference(I0, I1)
    if n == 1:
        return [middle]
    first_half = make_inference(mdl, I0, middle, n//2)
    second_half = make_inference(mdl, middle, I1, n//2)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def load_frame(imframe, pad_fn, device=device):
    t_img = torch.from_numpy(np.transpose(imframe, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    return pad_fn(t_img)


def sorted_image_list(image_dir, extension='.jpg'):
    frames = [f for f in os.listdir(
        image_dir) if f.endswith(f'{extension}')]
    frames.sort(key=lambda x: int(x[:-4]))
    return list(map(lambda x: os.path.join(image_dir, x), frames))


def main(argv):
    del argv    # Unused
    pyramid_scale = (2 ** FLAGS.exp) - 1

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if FLAGS.fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Model()
    model.load_model(FLAGS.model_dir, -1)
    print("Loaded v3.x HD model.")
    model.eval()
    model.device()

    videogen = sorted_image_list(FLAGS.image_dir, FLAGS.image_ext)
    pbar = tqdm(total=len(videogen))
    previous_frame = cv2.imread(videogen.pop(0), cv2.IMREAD_UNCHANGED)[
        :, :, ::-1].copy()
    h, w, padding = compute_dimensions(previous_frame)

    def _pad_fn(x):
        return F.pad(x, padding).half() if FLAGS.fp16 else F.pad(x, padding)

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)

    _thread.start_new_thread(
        build_read_buffer, (read_buffer, videogen))
    _thread.start_new_thread(
        clear_write_buffer, (write_buffer, FLAGS.output_dir))

    I1 = load_frame(previous_frame, _pad_fn)
    skip_frame = 1

    while True:
        frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = load_frame(frame, _pad_fn)
        I0_small = F.interpolate(
            I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(
            I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim > 0.995:
            if skip_frame % 100 == 0:
                print(f'\nWarning: Your video has {skip_frame} static frames, '
                      'skipping them may change the duration of the generated video.')
            skip_frame += 1
            if FLAGS.skip:
                pbar.update(1)
                continue

        if ssim < 0.2:
            output = [I0 for i in range(pyramid_scale)]
        else:
            output = make_inference(model, I0, I1, pyramid_scale)

        write_buffer.put(previous_frame)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        pbar.update(1)
        previous_frame = frame

    write_buffer.put(previous_frame)
    while(not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()


if __name__ == '__main__':
    app.run(main)
