import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        
        # CNN (feature extractor)
        # BatchNorm on every conv layer stabilises training, especially important for small datasets.
    
        self.cnn = nn.Sequential(
            # (1 -> 32,  H/2, W/2)
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # (32 -> 64, H/4, W/4)
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # (64 -> 128, H/8, W/8)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (128 -> 128, H/8, W/8)
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # keep width

            # (128 -> 256, H/16, W/8)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256 -> 256, H/16, W/8)
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # keep width

            # (256 -> 512, H/32, W/8)
            nn.Conv2d(256, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        
        # RNN (sequence modeling)
        
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )

        # Lowered from 0.3->0.1
        self.dropout = nn.Dropout(0.1)

        
        # CLASSIFIER
        
        self.fc = nn.Linear(512, num_classes)  # 256*2 (bidirectional)

    def forward(self, x):
        # x: (B, 1, H, W)
        conv = self.cnn(x)

        # conv shape: (B, C, H, W)
        b, c, h, w = conv.size()

        # CRNN expects height = 1
        assert h == 1, f"Expected height=1, got {h}"

        # reshape for RNN
        conv = conv.squeeze(2)        # (B, C, W)
        conv = conv.permute(2, 0, 1)  # (W, B, C)

        # RNN
        rnn_out, _ = self.rnn(conv)
        rnn_out = self.dropout(rnn_out)

        # FC
        output = self.fc(rnn_out)  # (T, B, num_classes)

        return output