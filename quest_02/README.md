# Quest 02. 거북이 미로찾기

>난이도: 🟡🟡🟡⚪⚪  
>장르: ColabTurtlePlus, 함수, 조건문 활용  

## AIFFEL Campus Online Code Peer Review

- 코더 : 김연
- 리뷰어 : 서승호

## PRT(Peer Review Template)

from turtle import Turtle, Screen

maze = [[0, 1, 0, 0, 0], 
        [0, 0, 0, 1, 0], 
        [0, 1, 1, 0, 0], 
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]]

start_x, start_y = 0, 0
end_x, end_y = 4, 4

window = Screen()
window.setup(width=300, height=300)

t = Turtle()
t.speed(1)
t.showturtle()

def solve_maze(x, y):
  if x == end_x and y == end_y:
    print("미로를 찾았습니다")
    return True

  if 0 <= x < 5 and 0 <= y < 5 and maze[y][x] == 0:
    maze[y][x] = 2  # 갔던 길
    t.goto(x * 10 + 5, y * 10 + 5)  # 거북이 다음 위치로 이동
    t.pendown()
    t.goto(x * 10 + 5, y * 10 + 5)
    t.penup()


    if solve_maze(x + 1, y) or solve_maze(x, y + 1) or solve_maze(
        x - 1, y) or solve_maze(x, y - 1):
      return True


    t.goto(x * 10 + 5, y * 10 + 5)
    t.pendown()
    t.goto(x * 10 + 5, y * 10 + 5)
    t.penup()
    maze[y][x] = 0  # 표시된 길 지우기

  return False


t.penup()
t.goto(start_x * 10 + 5, start_y * 10 + 5)
solve_maze(start_x, start_y)
window.update()
window.mainloop()


####### 회고 #######

# 배운 점 - 파이썬 거북이 게임의 원리에 대해 알 수 있었습니다.

# 아쉬운 점 - 리스트 함수형에 대한 이해와 연습이 더 필요해요.

# 느낀 점 - 문제를 풀이하기 위해 그루분과 다각도로 고민하고 자료를 찾는 과정이 유익했습니다.


-  ** 1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요? **
코드를 제출 하였으나, 전체 완성된 코드는 아닙니다.
그러나 완전히 작동하지 않는 것은 아니고, turtle이 움직이는 모습을 볼 수 있었기 때문에 조금만 수정하면 될 것 같습니다.       
-  ** 2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요? ** 주석도 조금 추가해 주시면 좋을 것 같습니다.   
-  ** 3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나”, ”새로운 시도 또는 추가 실험을 수행” 해봤나요? ** 네.   
-  ** 4. 회고를 잘 작성했나요? **
잘 작성하였습니다.    
-  ** 5. 코드가 간결하고 효율적인가요? **    
조금만 수정하면 좋을 것 같습니다.

## Review

저랑 다른 방식으로 푸는 코드를 볼 수 있어서 좋았습니다.  
