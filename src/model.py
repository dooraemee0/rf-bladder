import torch
import torch.nn as nn

class RFSingleEncoder(nn.Module):
    """
    1D RF 신호 하나를 인코딩하는 모듈.
    과적합 방지를 위해 드롭아웃을 추가하고, 두 종류의 풀링을 사용합니다.
    """
    def __init__(self, in_channels=1, out_dim=512, pool_output_size=10, dropout_p=0.25):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(pool_output_size)
        self.max_pool = nn.AdaptiveMaxPool1d(pool_output_size)

        self.fc = nn.Linear(64 * 2 * pool_output_size, out_dim)

    def forward(self, x):
        x = self.conv_block(x)
        pooled = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        flattened = pooled.flatten(start_dim=1)
        features = self.fc(flattened)
        return features


class RFStudent(nn.Module):
    """
    단순화된 RF Student 모델.
    - 각 RF 채널별 독립 인코더 사용
    - 어텐션, 프로젝터 제거
    - 모든 인코더의 특징을 연결하여 바로 예측 수행
    """
    def __init__(self, num_h_rf=3, num_s_rf=4, encoder_out_dim=512, dropout_p=0.5):
        super().__init__()
        self.num_h_rf = num_h_rf
        self.num_s_rf = num_s_rf

        # 각 채널별 독립 인코더 (드롭아웃 비율 전달)
        encoder_dropout = dropout_p / 2
        self.h_encoders = nn.ModuleList([
            RFSingleEncoder(out_dim=encoder_out_dim, dropout_p=encoder_dropout)
            for _ in range(num_h_rf)
        ])
        self.s_encoders = nn.ModuleList([
            RFSingleEncoder(out_dim=encoder_out_dim, dropout_p=encoder_dropout)
            for _ in range(num_s_rf)
        ])

        # 최종 회귀 헤드 입력 차원 계산:
        # 모든 인코더의 출력을 연결(concatenate)한 크기
        final_input_dim = (num_h_rf + num_s_rf) * encoder_out_dim

        self.head = nn.Sequential(
            nn.Linear(final_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1)
        )

    def forward(self, rf_h_list, rf_s_list):
        # 각 RF 신호를 개별 인코더에 통과
        h_feats_list = [self.h_encoders[i](rf_h_list[i]) for i in range(self.num_h_rf)]
        s_feats_list = [self.s_encoders[i](rf_s_list[i]) for i in range(self.num_s_rf)]

        # 모든 인코더의 특징 벡터를 하나로 연결
        combined_features = torch.cat(h_feats_list + s_feats_list, dim=1)

        # 최종 회귀 헤드를 통해 volume 예측
        volume = self.head(combined_features).squeeze(-1)

        # API 일관성을 위해 f_H, f_S 대신 None 반환
        return volume, None, None


if __name__ == '__main__':
    model = RFStudent(dropout_p=0.5)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    bs = 4
    # num_h_rf=4, num_s_rf=3 에 맞춰 dummy input 생성
    dummy_h = [torch.randn(bs, 1, 400) for _ in range(3)]
    dummy_s = [torch.randn(bs, 1, 400) for _ in range(4)]

    vol, _, _ = model(dummy_h, dummy_s)
    print("Output shapes:")
    print(f"Volume: {vol.shape}")
