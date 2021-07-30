# UltimateTicTacToe

## 문제발견(동기 및 필요성)
AlphaGo Zero의 원리는 바둑뿐만 아니라 다른 완전정보게임에서도 쓸 수 있는 알고리즘이지만 현재 AlphaGo Zero의 원리를 다른 게임에서 시도해본 연구는 거의 없음. 게임 트리 복잡도가 10^50 정도인 완전정보게임 Ultiate Tic Tac Toe는 아직 풀리지 않은 게임이었고 AlphaGo Zero의 원리를 적용해 이 게임의 근사 최적해를 구해보고자 함.

## 연구설계(연구환경, 연구방법, 절차, 단계 등)
Ultimate Tic Tac Toe 인공지능 중에서 최고 성능을 보이던 기존의 단순 몬테카를로 트리 서치 알고리즘과 인공신경망을 활용해 몬테카를로 트리 서치를 개선시킨 알고리즘을 비교하고자 함. C++과 CUDA C로 직접 코딩해 CPU 멀티 쓰레딩과 GPU를 사용하는 2000줄이 넘는 코드를 완성함. 48개의 CPU 코어와 GPU(Titan Xp)를 사용하여 연구를 진행함.

## 연구수행(단계별 상세내용, 중간 결과 등)
기존 알고리즘에서 트리 서치를 1500만번 하도록 한 뒤에 셀프 대국으로 학습데이터를 만듬. 확률을 출력하는 Policy Network 인공신경망을 직접 디자인하고 48가지 경우로 나누어 실험해 가장 좋은 파라미터 값들을 결정함. Adam Optimizer로 인공신경망을 학습시킴. 트리 서치 과정에서 자식 노드를 만들 때 인공신경망 확률값이 낮으면 만들지 않게 함.

## 연구 결과(결과정리, 결과 및 고찰)
동등한 실력이면 무승부 84.4%, 선공이 13.1%, 후공이 2.5%로 이기는 것을 확인함. 기존의 알고리즘과 동일 시간으로 대국했고 새로운 알고리즘이 선공일 때는 (무승부, 선공 승, 후공 승) = (19%, 70%, 11%) 후공일 때는 (5%, 43%, 52%)를 확인함. 후공이 이기기 힘듬에도 기존 알고리즘을 상대로 높은 승률을 보임. AlphaGo Zero의 원리가 잘 적용되었음을 확인함.
 
## 연구를 통해 배우고 성장한 점
최적화가 매우 잘된 2000줄이 넘는 코드를 구현하면서 큰 틀에서 전체 코드를 디자인하고 모듈화하는 실력이 많이 늘었습니다. 코드가 더 길어도 모듈화를 잘 시킨다면 더 긴 코드도 구현할 수 있을 것 같은 느낌이 들었습니다. CUDA C를 책을 보며 공부하면서 병렬 처리에 대해 심도 있게 알게 되었고 AlphaGo Zero의 원리를 직접 구현해보면서 원리를 거의 완벽하게 이해하게 되었습니다.
이번 연구를 통해 모든 인공지능 연구가 사실은 매우 단순한 작업들(학습데이터 생성, 인공신경망 구조 설계, 파라미터값 결정, 성능 측정)으로 이루어져있다는 것을 알게 되었습니다. 현재 인공지능이 매우 각광받고 있지만 나중에는 엑셀보다 조금 복잡한 툴 정도가 되지 않을까 싶고 기본 실력이 있다면 언제든지 손쉽게 코딩하여 자신이 원하는 결과를 얻을 수 있을 것이라는 것을 알게 되었습니다.

2018.07.26.에 작성됨.
이후에 추가 연구를 통해 몇몇 연구 결과들이 추가될 수도 있음.
