import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

class StructuralStabilityDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, is_train=True):
        """
        초기 데이터 로드 및 파싱을 담당하는 Dataset 클래스
        csv_file에는 image_view1_path, image_view2_path, label 등의 정보가 있어야 합니다.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # TODO: 실제 제공되는 CSV 또는 메타데이터 형식에 맞게 수정 필요
        # self.data = pd.read_csv(csv_file)
        self.data = [None] * 100 # 임시 더미 리스트 (테스트용 100개)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # row = self.data.iloc[idx]
        # img1_path = os.path.join(self.data_dir, row['image_view1'])
        # img2_path = os.path.join(self.data_dir, row['image_view2'])
        
        # img1 = Image.open(img1_path).convert('RGB')
        # img2 = Image.open(img2_path).convert('RGB')
        
        # if self.transform:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)
            
        # label = row['label'] # 0: stable, 1: unstable
        # return img1, img2, torch.tensor(label, dtype=torch.float32)
        
        # 더미 데이터 반환 (실제 구현 전 테스트용)
        dummy_img = torch.randn(3, 224, 224)
        return dummy_img, dummy_img, torch.tensor(1.0)

def get_train_transform():
    """
    [핵심 포인트]
    학습 데이터셋은 '고정된 평가 환경'이므로, 평가/테스트 환경(무작위 조명, 카메라 시점)에
    대응할 수 있도록 Data Augmentation을 극대화해야 합니다.
    """
    return T.Compose([
        # 카메라 시점 변동에 대응하기 위한 회전, 원근법, 이동, 스케일 증강
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        
        # 조명 변동에 대응하기 위한 색상 증강
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_valid_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
