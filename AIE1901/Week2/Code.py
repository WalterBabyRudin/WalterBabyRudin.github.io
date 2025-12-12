#!/usr/bin/env python
# coding: utf-8

# ## Upload the Solver

# In[1]:


from coptpy import COPT


import pandas as pd
from coptpy import *

# Create COPT environment
env = Envr()

# === Create COPT model ===
model = env.createModel("m")


# ## Parameter

# In[2]:


T = 30 ## Finish within 30 days
N = 64 ## we have 64 locations, but we don't need to visit them all
K = 2 ## Only two kinds of resources: food and water
M = 1000 # a large number
L = 1200 # the weight restriction
J = 10000 # we have 10,000 dollars at the beginning
W = [0, 3, 2]  # weight for resource W[1] == water; W[2] == food
P = [0, 5, 10]  # price for resource P[1] == water; P[2] == food

comsumption = [{}, {"晴朗": 5, "高温": 8, "沙暴": 10}, {"晴朗": 7, "高温": 6, "沙暴": 10}]
## comsumption[1] == water; comsumption[2] == food


# In[3]:


# t -- day, i -- location, k -- resource
t_i = [(t, i) for t in range(1, T + 1) for i in range(1, N + 1)]  # 1 ~ T, 1 ~ N
t_k = [(t, k) for t in range(1, T + 1) for k in range(1, K + 1)]  # 1 ~ T, 1 ~ K
i_k = [(i, k) for i in range(1, N + 1) for k in range(1, K + 1)]  # 1 ~ N, 1 ~ K
t_0_i = [(t, i) for t in range(0, T + 1) for i in range(1, N + 1)]  # 0 ~ T, 1 ~ N
t_0_k = [(t, k) for t in range(0, T + 1) for k in range(1, K + 1)]  # 0 ~ T, 1 ~ K
t_i_k = [(t, i, k) for t in range(1, T + 1) for i in range(1, N + 1) for k in range(1, K + 1)]  
# 1 ~ T, 1 ~ N, 1 ~ K

## k=1: water; k=2: food


# In[4]:


import csv

def calculate_adjacent_hexagons():
    adjacent_map = {}
    for i in range(64):  # 原始编号0-63
        row = i // 8
        is_even_row = row % 2 == 0
        adjacent = []

        adjacent.append(i)  # 添加自身索引，保证对角线为1

        # 左右相邻（横向）
        if i % 8 != 0:
            adjacent.append(i - 1)
        if i % 8 != 7:
            adjacent.append(i + 1)

        # 上方相邻
        if row > 0:
            if is_even_row:
                if i % 8 != 0:
                    adjacent.append(i - 9)
                adjacent.append(i - 8)
            else:
                adjacent.append(i - 8)
                if i % 8 != 7:
                    adjacent.append(i - 7)

        # 下方相邻
        if row < 7:
            if is_even_row:
                adjacent.append(i + 8)
                if i % 8 != 0:
                    adjacent.append(i + 7)
            else:
                if i % 8 != 7:
                    adjacent.append(i + 9)
                adjacent.append(i + 8)

        # 过滤掉超出0-63范围的索引并去除重复值
        adjacent = list(set([x for x in adjacent if 0 <= x < 64]))
        adjacent_map[i] = adjacent
    return adjacent_map

# 生成邻接矩阵（第一行第一列为空，对角线为1）
def generate_adjacency_matrix():
    adjacent_map = calculate_adjacent_hexagons()
    matrix = []

    # 第一行：第一个单元格为空，其余为1-64
    header = [''] + [i + 1 for i in range(64)]
    matrix.append(header)

    # 生成每行数据：第一列为1-64，其余为0或1
    for i in range(64):
        row_data = [0] * 64
        for neighbor in adjacent_map[i]:
            row_data[neighbor] = 1
        # 行的第一个元素为实际编号（i+1），后面跟随连接关系
        matrix_row = [i + 1] + row_data
        matrix.append(matrix_row)

    return matrix

# 保存邻接矩阵到CSV文件（使用GBK编码）
def save_to_csv(matrix, file_path='adjacency_matrix.csv'):
    with open(file_path, 'w', newline='', encoding='gbk') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)


if __name__ == "__main__":
    adjacency_matrix = generate_adjacency_matrix()
    save_to_csv(adjacency_matrix)


# In[5]:


import pandas as pd

# Weather data for 30 days
weather_conditions = [
    '高温', '高温', '晴朗', '沙暴', '晴朗', '高温', '沙暴', '晴朗', '高温', '高温',
    '沙暴', '高温', '晴朗', '高温', '高温', '高温', '沙暴', '沙暴', '高温', '高温',
    '晴朗', '晴朗', '高温', '晴朗', '沙暴', '高温', '晴朗', '晴朗', '高温', '高温'
]

# Create DataFrame with days as columns and weather as the single row
# This structure allows weather_data.loc[0, str(t)] to work
df = pd.DataFrame([weather_conditions], columns=[str(i) for i in range(1, 31)])

# Save to CSV
df.to_csv('weather.csv', index=False, encoding='utf-8-sig')


# In[6]:


weather_data = pd.read_csv('weather.csv', encoding='utf-8')
pos_data = pd.read_csv('adjacency_matrix.csv', encoding='gbk')

# Special Location
village = {39: [1, 2], 62: [1, 2]} ## you can purchase the resource at village
mine_pos = {30: 1000, 55: 1000}
end_pos = [64]


# In[7]:


A = {(t, k): comsumption[k][weather_data.loc[0, str(t)]] for t, k in t_k}  
# base consumption for resource k on day t
B = {t: weather_data.loc[0, str(t)] == "沙暴" for t in range(1, T + 1)}  
# sandstorm weather
D = {(i, k): int(i in village.keys() and k in village[i]) for i, k in i_k}  
# binary - whether it is available to buy resource k at location i
E = {i: int(i in mine_pos) for i in range(1, N + 1)} 
# binary - whether location i have minery
F = {i: int(i in end_pos) for i in range(1, N + 1)}  
# binary - whether location i is the destination
G = {i: mine_pos[i] if i in mine_pos else 0 for i in range(1, N + 1)}  
# integer - the mining benefit at location i (=1000 if E[i]==1)
H = {(i, j): pos_data.loc[i - 1, str(j)] if i != j else 1 for i in range(1, N + 1) for j in range(1, N + 1)}  
# binary - whether location i and j are neighbour


# ## Decision Variable

# In[8]:


# Binary variable - whether we arrive at location i at day t
x = model.addVars(t_0_i, vtype=COPT.BINARY, nameprefix='x')

# Binary variable - whether the player stay at location i at day t
y = model.addVars(t_i, vtype=COPT.BINARY, nameprefix='y')

# Integer variable - the quantity of the purchased resource k at day t
z = model.addVars(t_0_k, vtype=COPT.INTEGER, nameprefix='z', lb=0)

# Binary variable - whether the player dig mine at day t
w = model.addVars([i for i in range(1, T + 1)], vtype=COPT.BINARY, nameprefix='w')


# In[9]:


# Binary variable - whether arrive at location with resource k at day t
a = model.addVars(t_0_k, vtype=COPT.BINARY, nameprefix='a')
model.addConstrs((a[(t, k)] == quicksum(x[(t, i)] * D[(i, k)] for i in range(1, N + 1)) for t, k in t_0_k),
                 r"a_{t,k}=\sum_{i=1}^{N}D_{i,k}")

# Binary variable - whether available to dig minery
b = model.addVars([i for i in range(1, T + 1)], vtype=COPT.BINARY, nameprefix='b')
model.addConstrs((b[t] == quicksum(y[(t, i)] * E[i] for i in range(1, N + 1)) for t in range(1, T + 1)),
                 r"b_t = \sum_{i=1}^{N}y_{t,i}E_i")

# Binary variable - whether the player arrived at destination
d = model.addVars([i for i in range(0, T + 1)], vtype=COPT.BINARY, nameprefix='d')
model.addConstrs((d[t] == quicksum(x[(t, i)] * F[i] for i in range(1, N + 1)) for t in range(0, T + 1)),
                 r"b_t = \sum_{i=1}^{N}x_{t,i}F_i")

# Continuous variable - the quantity of resource k at day t before consumption
u = model.addVars(t_0_k, vtype=COPT.INTEGER, nameprefix='u', lb=0)

# Continuous variable - the quantity of resource k at day t after consumption
v = model.addVars(t_0_k, vtype=COPT.INTEGER, nameprefix='v', lb=0)

# Continuous variable - the left fund at day t
s = model.addVars([i for i in range(0, T + 1)], vtype=COPT.INTEGER, nameprefix='s')


# ## Constraints

# In[10]:


# (1) Starting point
model.addConstr((x[(0, 1)] == 1), "（1）x_{0,1} = 1")

# (2) Must get back to destination before Day 30
model.addConstr((x[(T, N)] == 1), "（2）x_{T,N} = 1")

# (3) Must present at one location every day
model.addConstrs((x.sum(t, "*") == 1 for t in range(0, T + 1)), nameprefix=r"3 \sum_{i=1}^Nx_{t,i} = 1")
# x size = [(t, i) for t in range(0, T + 1) for i in range(1, N + 1)]
# for t in range(0, T + 1):
#     model.addConstr(x.sum(t, "*") == 1)

# (4) Can only visit adjacent regions
model.addConstrs(((x[(t, i)] + x[(t - 1, j)] <= H[(i, j)] + 1) for t, i in t_i for j in range(1, N + 1)),
                 nameprefix="4 x_{t,i} + x_{t-1,i} <= H_{i,j} + 1")


# (5) Weight limitation
model.addConstrs((quicksum(v[(t, k)] * W[k] for k in range(1, K + 1)) <= L for t in range(0, T + 1)),
                 nameprefix=r"6 \sum_{k=1}^{K}v_{t,k}*W_k <= L")

# (6) Stay at original place when sandstorm
model.addConstrs((y.sum(t, "*") == 1 for t in range(1, T + 1) if B[t]), nameprefix="7 stop")

# (7) Dig mine 
model.addConstrs((w[t] <= b[t] for t in range(1, T + 1)), nameprefix="8 w_t <= b_t")

# (8) Resource Purchase
model.addConstrs((z[(t, k)] <= a[(t, k)] * M for t in range(1, T + 1) for k in range(1, K + 1)),
                 nameprefix="9 z_{t,k} <= a_{t,k}*M")

# (9) No return after arrive at destination 
model.addConstrs((d[t] <= d[t + 1] for t in range(0, T)), nameprefix="10 d[t]<=d[t+1]")

# (10) Stay and moving constraints
model.addConstrs((y[(t, i)] <= x[(t, i)] for t, i in t_i), nameprefix="11.1 y_{t,i} <= x_{t,i}")
model.addConstrs((y[(t, i)] <= x[(t - 1, i)] for t, i in t_i), nameprefix="11.2 y_{t,i} <= x_{t-1,i}")
model.addConstrs((x[(t - 1, i)] + x[(t, i)] <= y[(t, i)] + 1 for t, i in t_i),
                 nameprefix="11.3 x_{t-1,i} + x_{t,i} <= y_{t,i} + 1")


# In[11]:


# 状态转移方程
# (1) 剩余资源状态转移
model.addConstrs((u[(t, k)] == v[(t - 1, k)] - \
                  (2 * w[t] - y.sum(t, "*") + 2 - d[t - 1]) * A[(t, k)] \
                  for t in range(1, T + 1) for k in range(1, K + 1)),
                 nameprefix=r"u_{t,k} = v_{t-1,k} - (2*w_t-\sum_{i=1}^Ny_{t,i} + 2 -  d_{t-1}) "
                            r"* \sum_{i=1}^{N}A_{t,i,k}*x_{t-1,i}")
model.addConstrs((v[(t, k)] == u[(t, k)] + z[(t, k)] for t, k in t_0_k), nameprefix='v_{t,k} = u_{t,k} + z_{t,k}')

# (2) 初始资源为0
model.addConstrs((u[(0, k)] == 0 for k in range(1, K + 1)), nameprefix="u_{0,k}=0")

# (3) 初始资金购买资源
model.addConstr((s[0] == J - quicksum(z[(0, k)] * P[k] for k in range(1, K + 1))),
                r"s_{0}=J-\sum_{k=1}^{K}z_{0,k}*P_k")

# (4) 资金转移方程
model.addConstrs((s[t] == s[t - 1] + 1000 * w[t] - 2 * quicksum(z[(t, k)] * P[k] for k in range(1, K + 1))
                  for t in range(1, T + 1)), nameprefix=r"s_{t} = s_{t-1} + G*w_t - 2*\sum_{k=1}^{K}z_{t,k}*P_{k}")


# ## Objective Function

# In[12]:


model.setObjective(s[T] + 0.5 * quicksum(v[(T, k)] * P[k] for k in range(1, K + 1)), COPT.MAXIMIZE)


# ## Optimal Solution

# In[13]:


model.solve()


# In[14]:


if model.status == COPT.OPTIMAL:
    print(f'The Maximum capital when arrived at destination: {model.objval}')
    print(f'Solving Time: {model.SolvingTime} seconds')


# In[15]:


def output_simple_movement(model):
    if model.status != COPT.OPTIMAL:
        return
    
    locations = {}
    daily_funds = {}
    for var in model.getVars():
        if var.name.startswith('x(') and abs(var.x) > 1e-6:
            try:
                indices = var.name[2:-1].split(',')
                day = int(indices[0])
                location = int(indices[1])
                locations[day] = location
            except:
                continue

        if var.name.startswith('s(') and abs(var.x) > 1e-6:
            try:
                day = int(var.name[2:-1].strip())
                amount = var.x
                daily_funds[day] = amount
            except:
                continue
    
    if locations:
        print("\nThe Optimal Path:")
        for day in sorted(locations.keys()):
            print(f"Day {day} → Region {locations[day]}: Left Capital {daily_funds[day]}")


# In[16]:


output_simple_movement(model)

