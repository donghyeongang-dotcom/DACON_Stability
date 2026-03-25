import torch
import torch.nn as nn
import torchvision.models as models

class DualViewPredictor(nn.Module):
    def __init__(self, pretrained=True):
        super(DualViewPredictor, self).__init__()
        
        # 백본 네트워크: ResNet50 등 활용 가능. 더 복잡한 물리 추론을 위해 ConvNeXt나 ViT도 고려해볼 수 있습니다.
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # ResNet의 마지막 FC 레이어를 제외한 특징 추출부
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # 2개의 시점을 병합(Concat)하므로 특징채널 사이즈는 두 배(2048 * 2 = 4096)
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # 평가셋이 완전히 다른 분포이므로 과적합 방지를 위한 드롭아웃
            nn.Linear(512, 1) # 이진 분류 (BCEWithLogitsLoss와 함께 사용)
        )

    def forward(self, view1, view2):
        # view1: (Batch, 3, 224, 224) -> feat1: (Batch, 2048, 1, 1)
        feat1 = self.features(view1).squeeze(-1).squeeze(-1)
        feat2 = self.features(view2).squeeze(-1).squeeze(-1)
        
        # 두 시점의 특징을 결합
        combined_features = torch.cat((feat1, feat2), dim=1)
        
        # 불안정성에 대한 로짓(Logit) 스코어 출력
        out = self.classifier(combined_features)
        return out
