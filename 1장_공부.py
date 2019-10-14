
# coding: utf-8

# # 1.1 CCTV 현황과 인구 현황 데이터 구하기
# * 서울시 자치구 연도별 CCTV 설치 현황 사이트
# * 서울시 서울통계 사이트

# # 1.2 파이썬에서 텍스트 파일과 엑셀 파일 읽기 -pandas

# In[40]:


# 모듈 불러오기
import pandas as pd


# In[41]:


# csv 파일 불러오기
CCTV_Seoul = pd.read_csv("../data/01. CCTV_in_Seoul.csv", encoding = "utf-8")
CCTV_Seoul.head()


# In[42]:


#CCTV_Seoul 데이터 끝 부분 확인
CCTV_Seoul.tail()


# In[43]:


#column(열) 확인
CCTV_Seoul.columns


# In[44]:


#column를 인덱스로 확인 가능
CCTV_Seoul.columns[0] #0:기관명, 1:소계, 2:2013년도 이전, 3:2014년, 4:2015년, 5:2016년


# In[45]:


#column의 이름 변경(rename)
CCTV_Seoul.rename(columns={CCTV_Seoul.columns[0]: "구별"}, inplace=True)#inplace=True:변수의 내용을 갱신해라
#{CCTV_Seoul.columns[0]:기관명 -> rename:구별
CCTV_Seoul.head()


# In[46]:


#excel 파일 불러오기
pop_Seoul = pd.read_excel("../data/01. population_in_Seoul.xls",encoding="utf-8")
pop_Seoul.head()


# In[47]:


#excel 파일 불러오기(옵션 적용)
pop_Seoul = pd.read_excel("../data/01. population_in_Seoul.xls",
                         header = 2, #세번째 줄부터 읽어라(파이썬은 0부터 시작한다)
                         usecols = "B, D, G, J, N", #엑셀의 B열, D열, G열, J열, N열만 골라서 불러와라(parse_cols->usecols)
                         encoding="utf-8")
pop_Seoul.head()


# In[48]:


#rename으로 이름 변경
pop_Seoul.rename(columns={pop_Seoul.columns[0] : "구별",
                         pop_Seoul.columns[1] : "인구수",
                         pop_Seoul.columns[2] : "한국인",
                         pop_Seoul.columns[3] : "외국인",
                         pop_Seoul.columns[4] : "고령자"}, inplace=True)
pop_Seoul.head()


# # 1-3 pandas 기초 익히기

# In[49]:


#모듈 불러오기
import pandas as pd
import numpy as np


# In[50]:


#pandas Series 사용(list 데이터)
s = pd.Series([1,3,5,np.nan,6,8]) #Series:list 데이터로 만든다
s


# In[51]:


#pandas data_range 사용(기본 날짜 설정)
dates = pd.date_range("20130101", periods=6)#periods=6 :2013년 01월 01일 부터 6일을 뽑아라
dates


# In[52]:


#데이터 프레임 생성
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=["A", "B", "C", "D"])
#np.random.randn(randn:가우시안 표준 정규 분포)
df


# In[53]:


#데이터프레임의 인덱스 보기
df.index


# In[54]:


#데이터 프레임의 columns 보기
df.columns


# In[55]:


#데이터 프레임의 내용물 보기(values)
df.values


# In[56]:


#데이터 프레임의 개요 보기(info())
df.info()


# In[57]:


#데이터 프레임의 통계적 개요 보기(describe())
df.describe() #values(내용물)이 숫자가 아니라 문자라도 그에 맞게 개요가 나타난다


# In[58]:


#sort_values(by=기준) 사용하기
df.sort_values(by="B", ascending=False)


# In[59]:


#데이터 프레임의 column으로 보기
df["A"]


# In[60]:


#데이터 프레임의 인덱스 0~3개 보기
df[0:3]


# In[61]:


#데이터 프레임의 특정 인덱스만 보기
df["20130102":"20130104"] # or df[1:3]


# In[62]:


#loc(location) 슬라이싱 사용(iloc와 차이점 알아두기!)
df.loc[dates[0]] #2013-01-01	-2.014368	0.540903	-1.209860	0.684123(index=dates, 가로로 추출)


# In[63]:


#loc로 a,b 행만 보기
df.loc[:,["A","B"]]#:=인덱스 모두


# In[64]:


#loc로 행과열 지정해서 보기
df.loc["20130102":"20130104",["A","B"]]


# In[65]:


#loc로 행과열 지정해서 보기2
df.loc["20130102",["A","B"]] # or df.loc[dates[1],["A","B"]]


# In[66]:


#iloc:loc 명령어와 달리 행과 열의 번호를 이용
df.iloc[3] #index[3],dates[3]의 뜻


# In[67]:


#iloc의 중요점!:왼쪽이 행,오른쪽이 열을 뜻한다(loc도 마찬가지, 다만 iloc는 번호를 이용한다!)
df.iloc[3:5,0:2]


# In[68]:


#iloc로 특정행과 열들을 골라서 보기
df.iloc[[1,2,4],[0,2]]


# In[69]:


#iloc로 전체범위 설정해서 보기
df.iloc[1:3,:]


# In[70]:


df.iloc[:,1:3]


# In[71]:


df


# In[72]:


#데이터프레임으로 조건으로 보기(True,False 형식)
df > 0 #True,False 형식으로 나옴


# In[73]:


#데이터프레임으로 조건으로 보기(리스트 형식)
df[df > 0]


# In[74]:


#데이터 프레임 복사하기(copy
df2 = df.copy()


# In[75]:


#데이터 프레임의 칼럼 추가하기
df2["E"] = ["one", "one", "two", "three", "four", "three"]
df2


# In[76]:


#데이터프레임의 특정 컬럼에 조건 걸기(isin)
df2["E"].isin(["four","three"])


# In[77]:


#데이터프레임의 특정 컬럼에 조건 걸기(isin) 리스트 형태
df2[df2["E"].isin(["four","three"])]


# In[78]:


#데이터 프레임에 특정 함수 적용(apply)
df.apply(np.cumsum) #누적합 함수


# In[79]:


#one-line 함수의 lambda
df.apply(lambda x: x.max() - x.min())


# # 1-4 pandas 이용해서 CCTV와 인구 현황 데이터 파악하기

# In[80]:


#데이터 확인 하기
CCTV_Seoul.head()


# In[81]:


#소계를 기준으로 내림차순 하기
CCTV_Seoul.sort_values(by="소계", ascending=True).head()
#도봉구,마포구,송파구,중랑구,중구 순으로 적은것을 확인


# In[82]:


#소계를 기준으로 오름차순 하기
CCTV_Seoul.sort_values(by="소계", ascending=False).head()
#강남구,양천구,서초구,은평구,용산구 순으로 많은것을 확인


# In[83]:


#최근증가율 컬럼을 만들고 그속에 최근 3년간 cctv증가율 계산한것을 넣기(2014+2015+2016 / 2013)
CCTV_Seoul["최근증가율"] = (CCTV_Seoul["2016년"] + CCTV_Seoul["2015년"] + CCTV_Seoul["2014년"]) / CCTV_Seoul["2013년도 이전"] * 100
CCTV_Seoul.sort_values(by="최근증가율", ascending=False)


# In[84]:


#서울시 데이터 확인
pop_Seoul.head()


# In[85]:


#필요없는 행 지우기(drop)
pop_Seoul.drop([0], inplace=True)
pop_Seoul


# In[86]:


#unique 칼럼 보기(중복 데이터 거르기)
pop_Seoul["구별"].unique()
#nan값 확인


# In[87]:


#nan값이 어디 있는지 확인하기(isnull)
pop_Seoul[pop_Seoul["구별"].isnull()]


# In[88]:


#nan값 있는 행 삭제하기
pop_Seoul.drop([26], inplace=True)


# In[89]:


pop_Seoul.tail()


# In[90]:


#전체 인구로 외국인 비율과 고령자 비율 계산후 칼럼 만들기
pop_Seoul["외국인비율"] = (pop_Seoul["외국인"] / pop_Seoul["인구수"]) * 100
pop_Seoul["고령자비율"] = (pop_Seoul["고령자"] / pop_Seoul["인구수"]) * 100
pop_Seoul.head()


# In[91]:


#인구수 기준으로 정렬하기
pop_Seoul.sort_values(by="인구수",ascending=False).head()


# In[92]:


#외국인 기준으로 정렬하기
pop_Seoul.sort_values(by="외국인",ascending=False).head()


# In[93]:


#외국인비율 기준으로 정렬하기
pop_Seoul.sort_values(by="외국인비율",ascending=False).head()


# In[94]:


#고령자 기준으로 정렬하기
pop_Seoul.sort_values(by="고령자",ascending=False).head()


# In[95]:


#고령자비율 기준으로 정렬하기
pop_Seoul.sort_values(by="고령자비율",ascending=False).head()


# # 1.5 pandas의 고급기능 두 데이터 병합하기

# In[96]:


#연습용 데이터 프레임 3개 만들기
df1 = pd.DataFrame({"A":["A0","A1","A2","A3"],
                    "B":["B0","B1","B2","B3"],
                    "C":["C0","C1","C2","C3"],
                    "D":["D0","D1","D2","D3"]},index=[0,1,2,3])
df2 = pd.DataFrame({"A":["A4","A5","A6","A7"],
                    "B":["B4","B5","B6","B7"],
                    "C":["C4","C5","C6","C7"],
                    "D":["D4","D5","D6","D7"]},index=[4,5,6,7])
df3 = pd.DataFrame({"A":["A8","A9","A10","A11"],
                    "B":["B8","B9","B10","B11"],
                    "C":["C8","C9","C10","C11"],
                    "D":["D8","D9","D10","D11"]},index=[8,9,10,11])


# In[97]:


#df1 데이터프레임 확인
df1.head()


# In[98]:


#df2 데이터프레임 확인
df2.head()


# In[99]:


#df3 데이터프레임 확인
df3.head()


# In[100]:


#데이터를 열방향으로 합치기(concat)
result = pd.concat([df1, df2, df3])
result


# In[101]:


#concat에 option(keys)
result = pd.concat([df1, df2, df3], keys = ["x", "y", "z"]) #keys = 다중 index으로 설정되어 level을 형성
result


# In[102]:


#result의 인덱스 확인
result.index


# In[103]:


#result index에서 level values(0)을 확인
result.index.get_level_values(0)


# In[104]:


#result index에서 level values(1)을 확인
result.index.get_level_values(1)


# In[105]:


#df4 생성하고 df1과 df4를 axis(1)로 concat하기
df4 = pd.DataFrame({"B":["B2","B3","B6","B7"],
                   "D":["D2","D3","D6","D7"],
                   "F":["F2","F3","F6","F7"]},index = [2, 3, 6, 7])
result = pd.concat([df1, df4], axis=1)


# In[106]:


#df1 확인
df1


# In[107]:


#df4 확인
df4


# In[108]:


#axis(1)한 result 확인
result #df4:index = [2, 3, 6, 7]
#concat 명령어는 index 기준으로 합친다는 것을 알수 있다.
#값을 가질수 없는 곳은 null값으로 처리된다 -> null 값 처리 대신 버리게 하는것:join="inner"


# In[109]:


#공통되지 않는 index의 데이터는 버리도록 하는 옵션(join="inner")
result = pd.concat([df1, df4], axis=1, join="inner")


# In[110]:


result
#문제점:null값 있는 곳을 삭제했더니 데이터가 있었던 곳도 전부 삭제되었다. 


# In[111]:


#df1의 인덱스에 맞추도록 하자(join_axes=[df1.index])
result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])
result
#문제점:df4의 나머지 인덱스은 합쳐지지 않고 삭제되었다.


# In[112]:


#열을 기준으로 합치는데 index를 무시하고 병합하자(ignore_index=True)
result = pd.concat([df1, df4], ignore_index=True)
result


# In[113]:


#left, right 데이터 두개 만들기
left = pd.DataFrame({"key":["K0", "K4", "K2", "K3"],
                    "A":["A0","A1","A2","A3"],
                    "B":["B0","B1","B2","B3"]})
right = pd.DataFrame({"key":["K0", "K1", "K2", "K3"],
                    "C":["C0","C1","C2","C3"],
                    "D":["D0","D1","D2","D3"]})


# In[114]:


#left 데이터 확인
left


# In[115]:


#right 데이터 확인
right


# In[116]:


#merge 명령어로 병합하기(on 옵션)
pd.merge(left, right, on = "key")
#key 열 기준으로 공통된것만 합친다(나머지는 삭제)


# In[117]:


#merge 명령어로 병합하기(how 옵션) ->left를 기준으로 설정
pd.merge(left, right, how="left", on = "key")

#left 데이터
#0	K0	A0	B0
#1	K4	A1	B1
#2	K2	A2	B2
#3	K3	A3	B3

#right 데이터
#0	K0	C0	D0
#1	K1	C1	D1
#2	K2	C2	D2
#3	K3	C3	D3


# In[118]:


#merge 명령어로 병합하기(how 옵션) -> right를 기준으로 설정
pd.merge(left, right, how="right", on = "key")

#left 데이터
#0	K0	A0	B0
#1	K4	A1	B1
#2	K2	A2	B2
#3	K3	A3	B3

#right 데이터
#0	K0	C0	D0
#1	K1	C1	D1
#2	K2	C2	D2
#3	K3	C3	D3


# In[119]:


#merge의 합집합(how="outer")
pd.merge(left, right, how="outer", on = "key")


# In[120]:


#merge의 교집합(how="inner")
pd.merge(left, right, how="inner", on = "key") 


# # 1-6 CCTV 데이터와 인구 현황 데이터를 합치고 분석하기

# In[121]:


#merge 명령으로 두데이터 합치기
data_result = pd.merge(CCTV_Seoul, pop_Seoul, on="구별")
data_result.head()


# In[122]:


#의미 없는 칼럼 삭제(drop[행],del[열])
del data_result["2013년도 이전"]
del data_result["2014년"]
del data_result["2015년"]
del data_result["2016년"]
data_result.head()


# In[123]:


#칼럼중 하나("구별")를 인덱스로 설정(set_index)
data_result.set_index("구별", inplace=True)
data_result


# ## 상관관계 분석(np.corrcoef)
# * 0.1 이하:무시
# * 0.3 이하:약한 상관관계
# * 0.7 이하:뚜렷한 상관관계
# * 결과는 행렬로 나타난다, 주 대각선을 기준으로 대칭인 행렬이고 대각선을 빼고 다른 값을 읽으면 된다.

# In[124]:


#고령자 비율과 소계의 상관관계 분석
np.corrcoef(data_result["고령자비율"],data_result["소계"])


# In[125]:


#외국인 비율과 소계의 상관관계 분석
np.corrcoef(data_result["외국인비율"],data_result["소계"])


# In[126]:


#인구수와 소계의 상관관계 분석
np.corrcoef(data_result["인구수"],data_result["소계"])
#0.3 이하 이므로 약한 상관관계라는것을 알수 있다.


# In[127]:


#소계를 기준으로 내림차순 하기(sort_values)
data_result.sort_values(by="소계", ascending=False).head()


# # 1-7 파이썬의 대표 시각화 도구-matplotlib

# In[131]:


#모듈 불러오기
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
#결과를 바로 보여주기(plt.show()를 안써도 된다)


# In[130]:


#간단한 그래프 생성
plt.figure
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
#plt.show()


# In[132]:


#arange와 sin 사용
t = np.arange(0,12,0.01) #0~12까지 0.01 간격으로 생성
y = np.sin(t)


# In[133]:


#sin 함수 그래프 그리기
plt.figure(figsize=(10,6))
plt.plot(t, y)
#plt.show()


# In[138]:


#sin 함수 그래프 그리기(label,grid,title 옵션 적용)
plt.figure(figsize=(10,6))
plt.plot(t, y)
plt.grid()
plt.xlabel("time")
plt.ylabel("Amplitude(진폭)")
plt.title("Example of sinewave")
#plt.show()


# ## 그래프 한글 인코딩 코드

# In[137]:


import platform
from matplotlib import font_manager, rc

path = "c:/Windows/Fonts/malgun.ttf"
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')

plt.rcParams["axes.unicode_minus"] =False 


# In[139]:


#plot 두개를 한 화면 생성(legend(범례) 옵션 적용)
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t), label="sin")
plt.plot(t, np.cos(t), label="cos")
plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Example of sinewave")
#plt.show()


# In[140]:


#lw(선의 굵기),color(색상 지정)
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t), lw=3, label="sin")
plt.plot(t, np.cos(t), "r", label="cos") #r:red -> color="red"
plt.grid()
plt.legend()
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.title("Example of sinewave")
#plt.show()


# In[146]:


#임의의 두 데이터 생성후 그래프 그리기
t = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 5, 8, 9, 5, 3]
plt.figure(figsize=(10,6))
plt.plot(t, y, color="green")
#plt.show()


# In[147]:


#line style 지정
plt.figure(figsize=(10,6))
plt.plot(t,y, color="green", linestyle="dashed") 
#linestyles = ['-', '--', '-.', ':']
 #('solid', 'solid')      # Same as (0, ()) or '-'
 #('dotted', 'dotted')    # Same as (0, (1, 1)) or '.'
 #('dashed', 'dashed')    # Same as '--'
 #('dashdot', 'dashdot')
#plt.show()


# In[149]:


#marker 옵션 지정
plt.figure(figsize=(10,6))
plt.plot(t, y, color="green", linestyle="dashed", marker="o")
#marker 표시명령어 = https://matplotlib.org/3.1.1/api/markers_api.html
#plt.show()


# In[155]:


#marker 색상과 크기 지정(markerfacecolor, markersize)
plt.figure(figsize=(10,6))
plt.plot(t, y, color='green', linestyle='dashed', marker='o',
        markerfacecolor = 'blue', markersize=12)
plt.xlim([-0.5, 6.5])
plt.ylim([0.5, 9.5])
#plt.show()


# In[156]:


#scatter(흩어지게 하다)로 그래프 그리기

#데이터 생성
t = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])

#scatter 그래프 그리기
plt.figure(figsize=(10,6))
plt.scatter(t,y)
#plt.show()


# In[157]:


#scatter의 marker 지정
plt.figure(figsize=(10,6))
plt.scatter(t,y, marker=">")
#plt.show()


# In[158]:


#축 값에 따라 색상을 바꾸는 color map 지정
colormap = t

plt.figure(figsize=(10,6))
plt.scatter(t,y, s = 50, c = colormap, marker=">")#s = 50 marker 사이즈
plt.colorbar()
#plt.show()


# In[159]:


#numpy의 랜덤변수 함수로 데이터 세개 생성
s1 = np.random.normal(loc=0, scale=1, size=1000)
s2 = np.random.normal(loc=5, scale=0.5, size=1000)
s3 = np.random.normal(loc=10, scale=2, size=1000)


# In[160]:


#랜덤변수 데이터 그래프 그리기
plt.figure(figsize=(10,6))
plt.plot(s1, label="s1")
plt.plot(s2, label="s2")
plt.plot(s3, label="s3")
plt.legend()
#plt.show()


# In[161]:


#boxplot으로 그래프 그리기
plt.figure(figsize=(10,6))
plt.boxplot((s1, s2, s3))
plt.grid()
#plt.show()


# # 1-8 CCTV 현황 그래프로 분석하기

# In[162]:


#데이터 확인
data_result.head()


# In[163]:


#plot 붙여서 그래프 바로 그리기
data_result["소계"].plot(kind="barh", grid=True, figsize=(10,10)) #barh:수평방향, bar:수직방향
#plt.show()


# In[164]:


#plot 붙여서 정렬하고 수평 그래프 바로 그리기
data_result["소계"].sort_values().plot(kind="barh", grid=True, figsize=(10,10)) #barh:수평방향, bar:수직방향
#plt.show()


# In[165]:


#CCTV 비율 계산해서 칼럼 만들고 그래프 그리기
data_result["CCTV비율"] = data_result["소계"] / data_result["인구수"] * 100
data_result["CCTV비율"].sort_values().plot(kind="barh", grid=True, figsize=(10,10))
#plt.show()


# In[166]:


#scatter로 그래프 그리기
plt.figure(figsize=(6,6))
plt.scatter(data_result["인구수"], data_result["소계"], s=50)
plt.xlabel("인구수")
plt.ylabel("CCTV")
plt.grid()
#plt.show()


# In[167]:


#데이터를 대표하는 직선을 그리기
fp1 = np.polyfit(data_result["인구수"], data_result["소계"], 1) #세번쨰 인자는 찾고자 하는 함수의 차수입니다. 2를 넣어주면 2차식의 계수를 찾아달라
fp1


# In[168]:


#직선을 그리기 위해 x축과 y축 데이터 얻기
f1 = np.poly1d(fp1) #x축
fx = np.linspace(100000, 700000, 100) #y축
#자세한 설명은 https://pinkwink.kr/1127 사이트 참고


# In[169]:


plt.figure(figsize=(10,10))
plt.scatter(data_result["인구수"], data_result["소계"], s=50)
plt.plot(fx, f1(fx), ls="dashed", lw=3, color="g")
plt.xlabel("인구수")
plt.ylabel("CCTV")
plt.grid()
#plt.show()


# In[170]:


#두가지를 추가
 #1. 직선이 전체 데이터의 대표값 역할(인구수가 300000 일때 cctv는 1100정도)이면 그 경향에서 멀리 떨어진 구는 이름이 같이 나타나도록 한다.
 #2. 직선에서 멀어질수록 다른 색을 나타내도록 한다

#1.오차를 계산할수 있는 코드 생성후 오차가 큰 순으로 데이터를 정렬

#데이터를 대표하는 직선을 그리기
fp1 = np.polyfit(data_result["인구수"], data_result["소계"],1)

#직선을 그리기 위해 x축과 y축 데이터 얻기
f1 = np.poly1d(fp1)
fx = np.linspace(100000, 700000, 100)

#오차 칼럼 만들기
data_result["오차"] = np.abs(data_result["소계"] - f1(data_result["인구수"]))#np.abs:절대값을 구하는 함수

#오차를 기준으로 정렬하고 df_sort 데이터 생성
df_sort = data_result.sort_values(by="소계", ascending=False)
df_sort.head()


# In[171]:


#2. 텍스트와 color map

plt.figure(figsize=(14,10))
plt.scatter(data_result["인구수"],data_result["소계"],
           c=data_result["오차"], s=50)
plt.plot(fx, f1(fx), ls="dashed", lw=3, color="g")

for n in range(10):
    plt.text(df_sort["인구수"][n]*1.02, df_sort["소계"][n]*0.98,
            df_sort.index[n], fontsize=15)
    
plt.xlabel("인구수")
plt.ylabel("인구당비율")

plt.colorbar()
plt.grid()
#plt.show()

