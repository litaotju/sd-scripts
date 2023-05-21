import os
import cv2
import argparse
import glob
import torch
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

class FaceFixer():

    def __init__(self, bg_upsampler_:str, face_upsample:bool, upscale:int, detection_model:str) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bg_upsampler, self.face_upsampler = FaceFixer.create_upsampler(bg_upsampler_, face_upsample)
        self.net, self.face_helper = FaceFixer.create_net_and_face_helper(upscale, detection_model, self.device)

    @staticmethod
    def create_default_fixer():
        return FaceFixer(None, False, 1, "retinaface_resnet50")

    @staticmethod
    def set_realesrgan():
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer

        use_half = False
        if torch.cuda.is_available(): # set False in CPU/MPS mode
            no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
            if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
                use_half = True

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
            model=model,
            tile=args.bg_tile,
            tile_pad=40,
            pre_pad=0,
            half=use_half
        )

        if not gpu_is_available():  # CPU
            import warnings
            warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                            'The unoptimized RealESRGAN is slow on CPU. '
                            'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                            category=RuntimeWarning)
        return upsampler


    @staticmethod
    def create_upsampler(bg_upsampler_, face_upsample):
        # ------------------ set up background upsampler ------------------
        if bg_upsampler_ == 'realesrgan':
            bg_upsampler = FaceFixer.set_realesrgan()
        else:
            bg_upsampler = None

        # ------------------ set up face upsampler ------------------
        if face_upsample:
            if bg_upsampler is not None:
                face_upsampler = bg_upsampler
            else:
                face_upsampler = FaceFixer.set_realesrgan()
        else:
            face_upsampler = None
        return bg_upsampler, face_upsampler

    @staticmethod
    def create_net_and_face_helper(upscale, detection_model, device):
        # ------------------ set up CodeFormer restorer -------------------
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                connect_list=['32', '64', '128', '256']).to(device)

        # ckpt_path = 'weights/CodeFormer/codeformer.pth'
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model = detection_model,
            save_ext='png',
            use_parse=True,
            device=device)
        return net, face_helper

    def restore_face(self, 
                     img, fidelity_weight=0.5, only_center_face=False, upscale=1, face_upsample=False, draw_box=False):
        w = fidelity_weight

        # clean all the intermediate results to process the next image
        self.face_helper.clean_all()
        if isinstance(img, str):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else:
            assert isinstance(img, np.ndarray), f'Wrong input type: {type(img)}'

        self.face_helper.read_image(img)
        # get face landmarks for each face
        num_det_faces = self.face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
        # align and warp each face
        self.face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        # upsample the background
        if self.bg_upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = self.bg_upsampler.enhance(img, outscale=upscale)[0]
        else:
            bg_img = None
        self.face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        if face_upsample and self.face_upsampler is not None: 
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=self.face_upsampler)
        else:
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

        return restored_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/whole_imgs', 
            help='Input image, video or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
            help='Output folder. Default: results/<input_name>_<w>')
    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5, 
            help='Balance the quality and fidelity. Default: 0.5')
    parser.add_argument('-s', '--upscale', type=int, default=2, 
            help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')
    parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
    parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
    parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')

    args = parser.parse_args()
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        img_path = args.input_path
    else:
        raise ValueError(f'Wrong input path: {args.input_path}')

    img_name = os.path.basename(img_path)
    basename, ext = os.path.splitext(img_name)
    fixer = FaceFixer.create_default_fixer()

    restored_img = fixer.restore_face(img_path, args.fidelity_weight, args.only_center_face, 
                 args.upscale, args.face_upsample, args.draw_box)
    save_restore_path = os.path.join("results", 'final_results', f'{basename}.png')
    imwrite(restored_img, save_restore_path)
