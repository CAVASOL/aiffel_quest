# Quest 00. Avengers 2-gram

```

다음 조건을 확인하여 Avengers Script에서 워드 단위의 2-gram을 구하고,
Script에서 가장 빈도가 높은 2-gram 페어를 찾아라!
조건 : Avengers.txt 파일을 사용한다.
      모든 문자는 소문자로 변환한다.(string 다루는 메서드 참고)
      모든 기호는 제거(정규표현식 참고)한 후, 2-gram을 구한다.

```

- 난이도 : 🟡🟡🟡⚪⚪
- 장르 : 내장함수 String, Collections, n-gram

## Peer Review

- 코더 : 김연
- 리뷰어 : 정인호

```

import re
from collections import Counter

file = open("/content/sample_data/Avengers.txt", "r")

contents = file.read()

# 어벤저스 텍스트를 소문자로 변환하고 기호를 제거해!
def text(text):
  text = re.sub(r'[^\w\s]', ' ', text) # regex로 어벤저스 텍스트 안의 기호 제거
  text = text.lower() # 소문자로 변환
  return text # finally! 출력

# 정리된 어벤저스 텍스트로 n-gram을 만들어보자
def generate_ngram(text, n):
  aha = text.split() # 텍스트 분리!
  ngrams = [] # n-gram을 저장할 빈 리스트 생성
  for i in range(len(aha) - n + 1): # n-gram을 만들어 줄 제너레이터
    ngram = ' '.join(aha[i:i+n])
    ngrams.append(ngram)
  return ngrams

text = text(contents)

ngrams = generate_ngram(text, 2) # 2-gram 제너레이터로 만들어진 그람들

count = Counter(ngrams) # 그람들이 몇 개나 될까

most = count.most_common() # 빈도가 높은 그람들

print(most)

```

- [x] 1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
     > 네.

- [x] 2. 주석을 보고 작성자의 코드가 이해되었나요?
     > 네.

- [x] 3. 코드가 에러를 유발할 가능성이 없나요?
     > 네.

- [x] 4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?
     > 네.

- [x] 5. 코드가 간결한가요?
     > 네.

## Reviw

```
# 정리된 어벤저스 텍스트로 n-gram을 만들어보자
def generate_ngram(text, n):
  aha = text.split() # 텍스트 분리!
  ngrams = [] # n-gram을 저장할 빈 리스트 생성
  for i in range(len(aha) - n + 1): # n-gram을 만들어 줄 제너레이터
    ngram = ' '.join(aha[i:i+n])
    ngrams.append(ngram)
  return ngrams
```

전체적으로 코드가 깔끔하고 간결하게 작성되어 있고 제네레이터를 활용한 n-gram 생성 부분이 매우 인상적이었습니다.
