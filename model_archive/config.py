import torch
from torchvision import transforms
# from .utils import PadTo16
import glob
import os
from openai import OpenAI
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
import torch.nn.functional as F
from PIL import Image, ImageOps

class PadTo16:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            return F.pad(img, (0, pad_w, 0, pad_h))  # pad (left, right, top, bottom)
        else:
            raise TypeError("Input must be torch.Tensor")

class PILPadTo16:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            return ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
        else:
            raise TypeError("Input must be PIL.Image")
        
class Config:

    def __init__(self):
        self.get_config()
        
    def get_config(self):
        load_dotenv()

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
        self.handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
        self.line_channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        self.user_id = os.getenv("USER_ID")
        self.liff_id = os.getenv("LIFF_ID")
        self.web_url = os.getenv("WEB_URL")
        self.channel_id = os.getenv("CHANNEL_ID")
        self.db_path = os.getenv("DB_PATH")

        self.classes_dict = {"Background": 0, "Green": 1, "Yellow": 2, "Red": 3}
        self.class_color_map = [(0, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0)]
        self.index_to_classes_dict = {0: "Background", 1: "Green", 2: "Yellow", 3: "Red"}
        self.class_names = [cls_name for cls_name, _ in self.classes_dict.items()]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.ToTensor()

        self.image_transform = transforms.Compose([
            PILPadTo16(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            PadTo16(),
            transforms.Resize((512, 512))
        ])

        self.image_transform_dinov2 = transforms.Compose([
            PILPadTo16(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        self.mask_transform_dinov2 = transforms.Compose([
            PadTo16(),
            transforms.Resize((224, 224))
        ])

        self.image_transform_ema = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.mask_transform_ema = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.image_transform_moe = transforms.Compose([
            transforms.Resize((160, 224)),
            transforms.ToTensor()
        ])

        self.image_transform_cascade = transforms.Compose([
            transforms.ToTensor()
        ])

        # 收集所有圖與標註檔案
        self.root_dir = "model_archive/"
        # self.root_dir = "./"
        self.train_image_paths = sorted(glob.glob(self.root_dir + "dataset/train/images/*.png"))
        self.train_ann_paths = sorted(glob.glob(self.root_dir + "dataset/train/annotations/*.pt"))
        # self.train_ann_mask_paths = sorted(glob.glob(self.root_dir + "dataset/train/annotations/*.pt"))
        self.val_image_paths = sorted(glob.glob(self.root_dir + "dataset/val/images/*.png"))
        self.val_ann_paths = sorted(glob.glob(self.root_dir + "dataset/val/annotations/*.pt"))
        # self.val_ann_mask_paths = sorted(glob.glob(self.root_dir + "dataset/val/annotations/*.pt"))
        self.test_image_paths = sorted(glob.glob(self.root_dir + "dataset/test/images/*.png"))
        self.test_ann_paths = sorted(glob.glob(self.root_dir + "dataset/test/annotations/*.pt"))
        # self.test_ann_mask_paths = sorted(glob.glob(self.root_dir + "dataset/test/annotations/*.pt"))

        self.inference_image_paths = sorted(glob.glob("static/uploads/*.jpg"))
        self.inference_ann_paths = None
        self.inference_ann_mask_paths = None

        self.model_dir = self.root_dir + "checkpoints"
        self.img_size = (384, 512)
        self.resize_img_size = (384, 512)
        # self.resize_img_size = (160, 224)
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.batch_size = 4
        self.num_classes = 3
        self.save_dir = "static/test_preds"
        self.optimizer_type = "adam"
        self.scheduler_mode = "cosineanneal"

        self.total_epochs = 10
        self.mode = "train"
        self.model = "unetr_moe"
        self.model_tuning_enable = True
        self.tensorboard_mode_enable = True
        self.start_epoch = 1
        
