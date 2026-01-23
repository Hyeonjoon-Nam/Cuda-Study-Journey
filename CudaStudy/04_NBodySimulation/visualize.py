import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    try:
        # 1. CSV 데이터 읽기
        df = pd.read_csv('nbody_result.csv')
        print(f"데이터 로드 완료: {len(df)}개의 별")

        # 2. 3D 그래프 설정
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 배경 스타일 (우주 느낌)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # 축 색상 설정
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.grid(False)

        # 3. 산점도 그리기
        # 질량(w)이 클수록 점을 크게 그림
        sizes = df['w'] * 20 
        # 색상은 Cyan 계열
        ax.scatter(df['x'], df['y'], df['z'], s=sizes, c='cyan', alpha=0.6, edgecolors='none')

        # 4. 카메라 시점 및 범위 설정
        # 초기 랜덤 범위가 -1 ~ 1 이었지만, 200번 움직였으니 조금 퍼졌을 수 있음
        #limit = 1.5 
        #ax.set_xlim([-limit, limit])
        #ax.set_ylim([-limit, limit])
        #ax.set_zlim([-limit, limit])
        
        ax.set_title(f"N-Body Result (N={len(df)})", color='white', fontsize=15)
        plt.axis('off') # 축 눈금 끄기
        
        print("그래프 출력 중...")
        plt.show()

    except FileNotFoundError:
        print("오류: 'nbody_result.csv' 파일을 찾을 수 없습니다.")
        print("C++ 프로그램을 먼저 실행했는지 확인해주세요.")

if __name__ == "__main__":
    main()