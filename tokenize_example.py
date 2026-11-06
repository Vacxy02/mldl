from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = ["AI 기업의 과도한 투자와 부채 부담을 우려한 투자자들이 미 국채를 매수했고, 그 결과 국채금리가 내려갔습니다. 시장에서는 금리 하락을 주식시장에는 우호적인 신호로 받아들였습니다."]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

print("단어 사전:")
for word, idx in tokenizer.word_index.items():
    print(f"  {idx}: {word}")

sequences = tokenizer.texts_to_sequences(text)
print("\n정수 시퀀스:")
print(sequences)

padded = pad_sequences(sequences, padding="post")
print("\n패딩 후:")
print(padded)
print(f"\n총 단어 수: {len(tokenizer.word_index)}")
print(f"시퀀스 길이: {len(sequences[0])}")
