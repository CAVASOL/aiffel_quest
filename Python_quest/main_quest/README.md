# Main Quest 01. Python Module Main Quest

- 난이도 : 🟡🟡🟡🟡⚪  
- 장르 :  Python

## Retrospect

```
Q9. Aiffel_Guild 클래스에 길드별 점수를 입력하고 차감하는 guild_score 메서드를 추가해보세요. 
    guild_score 메서드는 길드 이름과 점수를 인자로 받고 길드별 스코어를 출력합니다. 
    guild_score 메서드로 점수가 입력되거나 차감되면, Aiffel 클래스에서 구현된 개별 그루의 score도 
    소속 길드에서 추가되거나 차감된 점수에 따라 별도로 증감.

10. Q10. Aiffel_Group 클래스에 group2guild_score 메서드를 추가하세요. 
    group2guild_score 메서드는 아래와 같이 작동해야 합니다.
    group2guild_score 메서드는 그룹이름과 그룹점수 2개의 인자를 받습니다.
    랜덤으로 그룹핑된 그루들이 소속된 그룹의 이름으로 점수를 받게 되면 각자 소속된 길드의 점수로 연동되어 계산.

```

> a. 그루들은 랜덤으로 길드, 그룹으로 그룹핑되고  
> b. 길드로 또는 그룹으로 점수를 획득하면 개인의 스코어에도 점수가 반영  
>   
> 예를 들어, 김연은 하사비스 길드와 상생 그룹에 소속되어 있는데 하사비스 길드나 상생 그룹이 점수를 얻거나 잃으면  
> 그 가감된 점수가 김연의 개인 스코어에 반영되는 시스템인 것. 그래서 그루가 획득한 각종 점수들이 총점에 반영될 수 있도록 디자인 할 것,  
> 이 부분을 이해하기까지 많은 시간을 할애했다 딕셔너리로 정보를 맵핑하고 저장된 정보를 디스플레이하는 것에 대한 이해가 더 필요하다.  
> 흥미로운 퀘스트이다 위의 9, 10번 문제에 대한 답을 반영해서 전체적으로 코드를 다시 디자인해야겠다.  
