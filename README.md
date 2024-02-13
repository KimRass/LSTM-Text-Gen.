# 1. Pre-trained Model
- [lstm-epicurious.ckpt](https://drive.google.com/file/d/1Idxm_gsohMesbELqRgjYDJRrDADguK9U/view?usp=sharing)
    - Trained for 49 epochs.
    - Validation loss: 1.848.
    - Not weights only.

# 2. Implementation Details
- 전체 데이터 중 10%를 Test set으로 지정하고 나머지 중 20%를 Validation set으로 지정했습니다.
- Hugging Face의 GPT-2 Tokenizer를 사용했습니다. Vocab size 50,257을 수정 없이 그대로 사용했으며 Padding token만 추가했습니다.
- 샘플링 시에는 Test set에 있는 Prefix ("The recipe for '...':")를 처음에 모델에 입력한 후 토큰을 하나씩 생성하도록 했습니다.
