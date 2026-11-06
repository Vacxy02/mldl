import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def main():
    # 1. 텍스트 벡터화 레이어 정의
    vectorize_layer = TextVectorization(
        max_tokens=1000,
        output_mode='int',
        output_sequence_length=25  # 시퀀스 길이를 고정 (모자라면 패딩, 넘치면 자름)
    )

    # 2. 대상 문장 정의
    text = "AI 기업의 과도한 투자와 부채 부담을 우려한 투자자들이 미 국채를 매수했고 그 결과 국채금리가 내려갔습니다 시장에서는 금리 하락을 주식시장에는 우호적인 신호로 받아들였습니다"

    # 3. 레이어에 어휘집(Vocabulary) 학습
    vectorize_layer.adapt([text])

    # 4. 토큰화 및 정수 인코딩 수행
    tokenized_output = vectorize_layer([text])

    # 5. 결과 출력
    print("=" * 60)
    print(f"입력 문장: '{text}'")
    print("-" * 60)
    print("생성된 단어 사전 (Vocabulary):")
    vocab = vectorize_layer.get_vocabulary()
    for idx, word in enumerate(vocab):
        print(f"  인덱스 {idx}: '{word}'")

    print("-" * 60)
    print("토큰화 정수 시퀀스 결과 (Shape: [1, 10]):")
    print(tokenized_output.numpy())
    print("=" * 60)

if __name__ == "__main__":
    main()
