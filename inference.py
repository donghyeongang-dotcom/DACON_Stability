import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm # 진행률을 보기 위한 라이브러리

from dataset import StructuralStabilityDataset, get_valid_transform
from model import DualViewPredictor

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 데이콘 제공 제출 양식 불러오기
    sample_sub_path = "sample_submission.csv"
    submission_df = pd.read_csv(sample_sub_path)
    
    # 2. 학습된 모델 로드
    # 추론할 때는 pretrained=False여도 상관없습니다. 가중치를 바로 덮어씌우기 때문입니다.
    model = DualViewPredictor(pretrained=False, share_weights=False).to(device)
    
    # 학습 때 저장한 best_model.pth 불러오기
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval() # 평가 모드 전환 (Dropout, BatchNorm 동작 정지)

    # 3. 테스트 데이터셋 및 데이터로더 설정
    # Test 데이터는 sample_submission.csv에 id 목록이 있으므로 이를 활용
    test_dataset = StructuralStabilityDataset(
        data_dir="test", 
        csv_file=sample_sub_path, 
        transform=get_valid_transform(), # 추론 시에는 원본 형태를 최대한 유지(Train용 Augmentation 사용 X)
        is_test=True # 라벨 대신 id를 반환하도록 설정
    )
    
    # 추론 시에는 데이터를 섞으면 안 됩니다 (shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    results = []

    # 4. 추론 루프 실행
    print("Starting inference...")
    with torch.no_grad():
        for front_img, top_img, folder_ids in tqdm(test_loader, desc="Predicting"):
            front_img = front_img.to(device)
            top_img = top_img.to(device)
            
            # AMP 적용 (Train과 동일하게 빠른 속도 보장)
            with torch.cuda.amp.autocast():
                # model.py에 만들어둔 확률 반환 헬퍼 함수 호출
                stable_probs, unstable_probs = model.get_probabilities(front_img, top_img)
            
            # GPU에 있는 텐서를 CPU로 내리고 numpy 배열로 변환
            stable_probs = stable_probs.cpu().numpy()
            unstable_probs = unstable_probs.cpu().numpy()
            
            # 예측 결과를 리스트에 저장
            for i in range(len(folder_ids)):
                results.append({
                    'id': folder_ids[i],
                    'unstable_prob': unstable_probs[i],
                    'stable_prob': stable_probs[i]
                })
                
    # 5. 결과 저장 및 제출 파일 생성
    results_df = pd.DataFrame(results)
    
    # 혹시 모를 id 순서 꼬임을 완벽 방지하기 위해 원본 제출 양식의 id를 기준으로 병합(Merge)
    submission_df = submission_df.drop(columns=['unstable_prob', 'stable_prob'])
    final_submission = pd.merge(submission_df, results_df, on='id', how='left')
    
    # 데이콘 제출 양식 컬럼 순서(id, unstable_prob, stable_prob) 복원
    final_submission = final_submission[['id', 'unstable_prob', 'stable_prob']]
    
    # 최종 제출 파일 저장
    output_filename = "my_submission_v1.csv"
    final_submission.to_csv(output_filename, index=False)
    print(f"\n>>> Inference complete! Saved to {output_filename}")
    print(">>> 이제 데이콘 홈페이지에 접속해서 이 파일을 제출하시면 됩니다!")

if __name__ == "__main__":
    inference()
