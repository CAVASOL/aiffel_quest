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


# 회고 

# 배운 점 - n-gram 만들기를 배웠습니다  

# 아쉬운 점 - 메서드 검색이 필수였어요! python의 다양한 메서드를 알아두면 매우 유용한 것 같아요.  

# 느낀 점 - 초반에 어벤저스 텍스트 파일을 어디에서 다운로드 해야 하는지 헤맸지만 그루님의 도움 덕분에 파일을 받을 수 있었고,  
#         문제를 해결하기 위해 그루님과 다각도로 고민하고 자료를 찾아보는 과정이 매우 유익했습니다.  

