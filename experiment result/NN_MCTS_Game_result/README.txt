이 결과는 NN과 MCTS의 게임 플레이 결과이다.
cutting = 0.05: (12,44,7), (3,28,34)
cutting = 0.1: (11,37,17), (9,27,28)
cutting = 0.15 : (7,46,12), (7,27,19)
각각의 괄호쌍은 앞쪽 괄호쌍은 NN이 선공일 때의 결과, 뒤쪽 괄호쌍은 NN이 후공일 때의 결과이다.
(선공 WIN, DRAW, 선공 LOSS) 횟수를 기록.
매우 결과가 잘 나온 것을 볼 수 있고 NN+mcts가 mcts를 상대로 승리하는 것을 볼 수 있다.
아주 공정한 결과를 확인하기 위해 각각의 플레이어가 생각하는 시간을 동일하게 했다.
즉 동일 시간내에 판단해서 플레이를 하도록 했다. (9초로 실험함)
NN과 MCTS의 게임은 src/main.cu로 reproducing이 충분히 가능하다.

1. find_c_result
최적의 c값을 찾기 위해 c = 1이랑 c = 2
끼리 대국을 펼치도록 했다. 그 결과
(DRAW,ONE,TWO) = (27,1,2) 이 때는 c = 1이 선공
(DRAW,ONE,TWO) = (21,9,0) 이 때는 c = 2가 선공
즉 c = 2가 c = 1에 비해 더 잘하는 것을 알게됨
또한 선공이 더 유리하다는 것 또한 알게됨

2. c = 2끼리 대국을 시켜봄 (result001 ~ result060)
(DRAW,ONE,TWO) = (46,13,1)
DRAW_REWARD = 0으로 실험함
어느 정도 비율로 선공이 유리한지를 알게 됨.

3. DRAW_REWARD (result061 ~ result260)
최적의 DRAW_REWARD를 찾기 위해 대국을 시켜봄
(DRAW,ONE,TWO) = (74,21,5)
이 때는 선공이 DRAW_REWARD = -0.5,
후공이 DRAW_REWARD = 0인 결과
(DRAW,ONE,TWO) = (76,19,5)
이 때는 선공이 DRAW_REWARD = 0,
후공이 DRAW_REWARD = -0.5

이 실험까지의 중요한 실수일수도 있는 무언가를 발견
첫 문장 srand(...);에 주석처리 되어 있었음
그러나 fasrand()함수의 g_seed를 보아하니
별 문제 없을 수도...

4. search_time = 3에 대한 이유 증명(그러나 실패)
(result261~result340)
선공이 5초, 후공이 3초 고민하게 했을 때 결과는
(DRAW,ONE,TWO) = (29,21,0)

선공이 7초, 후공이 5초 고민하게 했을 때 결과는
(DRAW,ONE,TWO) = (13,7,0)

선공이 9초, 후공이 7초
(DRAW,ONE,TWO) = (15,5,0)
(나중에 추가된 결과 (result 411~result 420를 포함
하였다.)

선공이 11초, 후공이 9초
-> run time error 아마도 메모리 초과일듯
으로 인해 게임 진행 불가함

5. search_time = 9끼리 play하게 시킴
(result341~result390)
(result421~result520)
선공 = 후공 = 9초일 때의 결과
(DRAW,ONE,TWO) = (110,33,7)

6. search_time1 = 7, search_time2 = 9
(result 391~result 410)
(DRAW,ONE,TWO) = (16,3,1)