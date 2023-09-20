class Fish: # 물고기의 객체 속성 초기화
    def __init__(self, name: str, speed: int):
        self.name = name # 물고기의 이름
        self.speed = speed # 물고기의 속도

    def swimming(self):
       # "(물고기)*가 *의 속도로 수영한다"는 문장을 출력하는 함수
        return f"{self.name} is swimming at {self.speed} m/s"

# 리스트 안에 있는 물고기의 움직임을 문장으로 출력하는 제너레이터 함수
def fish_moving(fish_list):
    for f in fish_list:
        yield f.swimming()

# 니모와 도리가 있는 물고기 리스트
fish_list = [
    {"name": "Nemo", "speed": 3},
    {"name": "Dory", "speed": 5},
]

# 리스트 컴프리헨션을 사용, 딕셔너리 리스트인 fish_list에서 Fish클래스의 객체 생성
fish_list = [Fish(f["name"], f["speed"]) for f in fish_list]

# 컴프리헨션
print("Using Comprehension:")
fish_comp = [f.swimming() for f in fish_list]
for f in fish_comp:
    print(f)

# 제너레이터
print("Using Generator:")
fish_gen = fish_moving(fish_list)
for f in fish_gen:
    print(f)

    
