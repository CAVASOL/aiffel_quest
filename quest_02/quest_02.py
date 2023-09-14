# Quest 02. 다음은 ColabTurtle을 사용하여 미로를 찾는 문제. 조건을 확인하여 거북이가 미로를 헤매지 않도록 출구를 찾아주도록 하자.  

# 조건 : 미로는 5x5의 2차원 리스트로 주어진다. 시작 위치는 (0,0)이고 목적지는 (4,4)이다.  
# 터틀은 상하좌우로 움직일 수 있다. 미로 내에서 갈 수 있는 길은 0으로 표시되어 있다. 터틀이 이미 지나간 길은 2로 표시해야한다.  
# 거북이가 미로를 찾아 나가면 "미로를 찾았습니다.", 미로를 찾을 수 없으면 "미로를 찾을 수 없습니다."가 나올 수 있도록 만들어보자!  

# 주석으로 구현 내용이 써있는 부분을 코드로 채워주세요!!  
# 코드를 해석해주세요 부분은 코드에 대한 설명을 주석으로 써주세요!!  
# 결과물은 다음과 같아야 합니다 :)  


####### 답안 #######


from turtle import Turtle, Screen

maze = [[0, 1, 0, 0, 0], 
        [0, 0, 0, 1, 0], 
        [0, 1, 1, 0, 0], 
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]]

# 시작 위치와 목적지 위치
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

    # 다음 위치로 이동
    if solve_maze(x + 1, y) or solve_maze(x, y + 1) or solve_maze(
        x - 1, y) or solve_maze(x, y - 1):
      return True

    # 돌아가기
    t.goto(x * 10 + 5, y * 10 + 5)
    t.pendown()
    t.goto(x * 10 + 5, y * 10 + 5)
    t.penup()
    maze[y][x] = 0  # 표시된 길 지우기

  return False

# 시작 위치에서 미로 찾기 시작
t.penup()
t.goto(start_x * 10 + 5, start_y * 10 + 5)
solve_maze(start_x, start_y)
window.update()
window.mainloop()


####### 회고 #######


# 배운 점 - 파이썬 거북이 게임의 원리에 대해 알 수 있었습니다.

# 아쉬운 점 - 리스트 함수형에 대한 이해와 연습이 더 필요해요.

# 느낀 점 - 문제를 풀이하기 위해 그루분과 다각도로 고민하고 자료를 찾는 과정이 유익했습니다.


####### Ref #######


from turtle import Turtle, Screen

maze = [[0, 1, 0, 0, 0], 
        [0, 0, 0, 1, 0], 
        [0, 1, 1, 0, 0], 
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]]

start_x, start_y = 0, 0
end_x, end_y = 4, 4

window = Screen()
window.setup(width=100, height=100)

t = Turtle()
t.speed(1)

def solve_maze(x, y):
  if x == end_x and y == end_y:
    print("I found the maze")
    return True

  for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if 0 <= nx < 5 and 0 <= ny < 5 and maze[nx][ny] == 0:
          maze[nx][ny] = 2
          t.pendown()
          t.goto(nx*10, ny*10)
          t.penup()
  
          if solve_maze(nx, ny):
              return True
          else:
            t.pendown()
            t.goto(x*10, y*10)
            t.penup()
            maze[nx][ny] = 0

t.penup()
t.goto(start_x*10, start_y*10)
solve_maze(start_x, start_y)
window.update()
window.mainloop()
