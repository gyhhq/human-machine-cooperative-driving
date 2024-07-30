import os

import numpy as np
import math
import matplotlib.pyplot as plt
import Control.draw as draw
import torch
from scipy.special import ive
import warnings
import time
from prediction_trajectory import prediction_fake
from numpy.linalg import norm
warnings.filterwarnings("ignore",category=UserWarning)
from authority3 import func
from authority2 import func_d
from fz import fz_func
from authority1 import compute_rho_trajectory
import csv

# 定义常数
k = 0.1  # look forward gain
Lfc = 1.0  # look-ahead distance
Kp = 0.78 # speed propotional gain
Kd=0.22
dt = 0.1  # [s]
L = 2.454  # [m] wheel base of vehicle
W = 1.57 # [m] width of vehicle

#车辆相关参数
dc=0.4  #[m] 安全域度
alphaf=30 #调节势场代大小系数
rouy=1.1  #y方向的收敛系数

show_animation = True

class C:
    # PID config
    Kp = 0.3  # proportional gain
    # system config
    kf = 0.1  # look forward gain
    dt = 0.1 # T step
    dist_stop = 0.7  # stop distance
    dc = 0.0
    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 1.57  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.454 # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width
    MAX_STEER = 0.30
    MAX_ACCELERATION = 5.0
class VehicleState:  # 定义一个类，用于调用车辆状态信息
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((L / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((L / 2) * math.sin(self.yaw))

def update(state, a, delta):  # 更新车辆状态信息
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt
    state.rear_x = state.x - ((L / 2) * math.cos(state.yaw))
    state.rear_y = state.y - ((L / 2) * math.sin(state.yaw))
    return state

def PIDControl(target, current):  # PID控制，定速巡航
    a = Kp * (target - current)
    return a

def pure_pursuit_control(state, cx, cy, pind):  # 纯跟踪控制器
    ind = calc_target_index(state, cx, cy)  # 找到最近点的函数，输出最近点位置
    if pind >= ind:
        ind = pind
    if ind < len(cx):
        tx = cx[ind]
        ty = cy[ind]
    else:
        tx = cx[-1]
        ty = cy[-1]
        ind = len(cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
    if state.v < 0:  # 如果是倒车的话，就要反过来
        alpha = math.pi - alpha
    Lf = 7
    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)  # 核心计算公式
    rou_pp=2*math.sin(alpha)/Lf

    return delta, ind,rou_pp

# 误差计算
def error_calculation(state,cx,cy):
    dx = [state.rear_x - icx for icx in cx]
    dy = [state.rear_y - icy for icy in cy]
    d = np.hypot(dx, dy)
    ind = np.argmin(d)
    d_d1 = np.hypot((state.rear_x - cx[ind]), (state.rear_y - cy[ind]))  # 当前车辆与轨迹最近位置的偏差
    return d_d1

def calc_target_index(state, cx, cy):
    dx = [state.rear_x - icx for icx in cx]
    dy = [state.rear_y - icy for icy in cy]
    d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))
    L = 0.0
    Lf = 7
    # search look ahead target point index
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind + 1] - cx[ind]
        dy = cx[ind + 1] - cx[ind]
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    return ind

class env:
    def __init__(self):
        self.road_width = 3.5/2
        self.bound_up = self.design_bound_up()
        self.bound_down = self.design_bound_down()
    def design_bound_up(self):
        bx_up, by_up = [], []
        for i in np.arange(0.0, 280, 0.1):
            bx_up.append(i)
            by_up.append(self.road_width+7+1.75)
        return bx_up, by_up
    def design_bound_down(self):
        bx_down, by_down = [], []
        for i in np.arange(0.0, 280, 0.1):
            bx_down.append(i)
            by_down.append(-self.road_width+3.5+1.75)
        return bx_down, by_down

def calculate_D(y1,y2):
    return abs(y2-y1)

#计算动态势场
#计算动态势场
def dpf_velocity(delta_v, delta_x,a,V):

    alpha_f =30 ;sigma_x = 10;b = 2
    eta = np.sign(delta_v) * np.abs(delta_v)
    delta_x = np.sign(-delta_x) * np.abs(delta_x)
    a = np.sign(a) * np.abs(a)
    # 计算 theta
    theta = np.pi * (1 + np.min([-np.sign(delta_x), 0]))
    # 计算 U_eo sudu
    I0 = np.i0(eta)
    I = ive(1, eta)
    U_eo = alpha_f * np.exp(eta * np.cos(theta) - I) / (2 * np.pi * I0)
    # 计算 U_rd  juli
    A_f = 2
    U_rd = A_f * np.exp(- (delta_x ** (2 * b)) / (2 * (sigma_x ** (2 * b))))
    # 计算加速度场
    T, R, K, G, thegma, tao, m = 1, 1, 20, 0.5, 1.2, 2, 1298.9  # 修改tao值能获得更远的测试距
    M = m * ((1.566 * 10 ** (-14)) + 0.3354)
    r0 = ((((abs(delta_x)) * (tao / np.exp(0.029 * V))) ** 2)) ** 0.5
    U1 = (M * T * R * K * G / ((K + a * math.cos(math.radians(theta))) * ((r0) ** thegma)))  # * ((r0) ** thegma)
    # 计算DPF场
    velocity_field = U_eo * U_rd * U1
    return velocity_field

#APF force calculation
def apf_force(y1,y2):  #y1是障碍物，y2是目标车辆
    rb=y2-y1
    F = alphaf * np.exp(-(rb ** 2) / (2 * rouy ** 2))
    U=F
    return U

def intent(a,b):
    # Find the minimum value in the augmented vector [a, b]
    a=torch.squeeze(a,dim=1)
    augmented = np.concatenate((a, b))
    k = np.min(augmented,axis=0)
    # Subtract k from a and b
    a_minus_k = a - k
    b_minus_k = b - k
    # Flatten the arrays to 1D vectors for dot product calculation
    a_flat = a_minus_k.flatten()
    b_flat = b_minus_k.flatten()
    # Compute the dot product
    dot_product = np.dot(a_flat, b_flat)
    # Compute the Euclidean norms
    norm_a_minus_k = np.linalg.norm(a_flat)
    norm_b_minus_k = np.linalg.norm(b_flat)
    # Compute rho
    rho = dot_product / (norm_a_minus_k * norm_b_minus_k)
    # Normalize delta_values
    min_val = 0.9981347   #按照最大转向角计算
    max_val = 1
    rho =  (rho - min_val) / (max_val - min_val)
    return rho

def main():
    #parameter
    ttime=0

    xy_target1_1=torch.zeros(1, 2);xy_target2_1=torch.zeros(1, 2)
    xy_target3_1=torch.zeros(1, 2);xy_target4_1=torch.zeros(1, 2)
    xy_target5_1 = torch.zeros(1, 2);xy_target6_1 = torch.zeros(1, 2)
    xy_target7_1 = torch.zeros(1, 2)

    xy_target1_2=torch.zeros(1, 2);xy_target2_2=torch.zeros(1, 2)
    xy_target3_2=torch.zeros(1, 2);xy_target4_2=torch.zeros(1, 2)
    xy_target5_2=torch.zeros(1, 2);xy_target6_2=torch.zeros(1, 2)
    xy_target7_2=torch.zeros(1, 2)

    bx1, by1 = env().design_bound_up()
    bx2, by2 = env().design_bound_down()
#绘制轨迹
    # 绘制轨迹
    cx1 = np.arange(0, 280, 0.01)
    cy1 = np.full_like(cx1, 3.5 + 1.75)
    # cx11 = np.arange(0, 200, 0.01)
    # cy11 = np.full_like(cx11, 7 + 1.75)

    cx2 = np.arange(0, 280, 0.01)  # 目标车辆轨迹
    # cy2 = [1.75+3.5+math.sin(ix / 1000.0) * (ix) / 2.0 for ix in cx2]
    cy2 = np.full_like(cx2, 3.5 + 1.75)
    cx3 = np.arange(0, 280, 0.01)
    cy3 = np.full_like(cx3, 3.5 + 1.75 )
    cx4 = np.arange(0, 280, 0.01)
    cy4 = np.full_like(cx4, 3.5 + 1.75)
    cx5 = np.arange(0, 280, 0.01)
    cy5 = np.full_like(cx4, 3.5 + 1.75)
    cx6 = np.arange(0, 280, 0.01)
    cy6 = np.full_like(cx4, 3.5 + 1.75)
    cx7 = np.arange(0, 280, 0.01)
    cy7 = np.full_like(cx4, 3.5 + 1.75)



    # 设定速度和时间
    target_speed1 = 40 / 3.6
    # target_speed11 = 20 / 3.6
    T1 = 700.0  # max simulation time
    target_speed2 = 50 / 3.6
    target_speed3 = 50 / 3.6
    target_speed4 = 50 / 3.6
    target_speed5 = 50 / 3.6
    target_speed6 = 50 / 3.6
    target_speed7 = 50 / 3.6

    # 初始化各参数
    state1 = VehicleState(x=40, y=3.5+ 1.75, yaw=0.0, v=10.0)
    lastIndex1 = len(cx1) - 1
    time1 = 0.0
    x1 = [state1.x]
    y1 = [state1.y]
    yaw1 = [state1.yaw]
    v1 = [state1.v]
    t1 = [0.0]
    target_ind1 = calc_target_index(state1, cx1, cy1)

    # state11 = VehicleState(x=45, y=7+ 1.75, yaw=0.0, v=10.0)
    # x11 = [state11.x]
    # y11 = [state11.y]
    # yaw11 = [state11.yaw]
    # v11 = [state11.v]
    # target_ind11 = calc_target_index(state11, cx11, cy11)


    state2 = VehicleState(x=00, y=3.5+ 1.75, yaw=0.0, v=10)   # human
    x2 = [state2.x]
    y2 = [state2.y]
    yaw2 = [state2.yaw]
    v2 = [state2.v]
    target_ind2 = calc_target_index(state2, cx2, cy2)

    state3 = VehicleState(x=00, y=3.5 + 1.75, yaw=0.0, v=10)  # cooperation
    x3 = [state3.x]
    y3 = [state3.y]
    yaw3 = [state3.yaw]
    v3 = [state3.v]
    target_ind3 = calc_target_index(state3, cx3, cy3)

    state4 = VehicleState(x=00, y=3.5 + 1.75, yaw=0.0, v=10)   # automation
    x4 = [state4.x]
    y4 = [state4.y]
    yaw4 = [state4.yaw]
    v4 = [state4.v]
    target_ind4 = calc_target_index(state4, cx4, cy4)

    state5 = VehicleState(x=00, y=3.5 + 1.75, yaw=0.0, v=10)   # automation
    x5 = [state5.x]
    y5 = [state5.y]
    yaw5 = [state5.yaw]
    v5 = [state5.v]
    target_ind5 = calc_target_index(state5, cx5, cy5)

    state6 = VehicleState(x=00, y=3.5 + 1.75, yaw=0.0, v=10)   # automation
    x6 = [state6.x]
    y6 = [state6.y]
    yaw6 = [state6.yaw]
    v6 = [state6.v]
    target_ind6 = calc_target_index(state6, cx6, cy6)

    state7 = VehicleState(x=00, y=3.5 + 1.75, yaw=0.0, v=10)   # automation
    x7 = [state7.x]
    y7 = [state7.y]
    yaw7 = [state7.yaw]
    v7 = [state7.v]
    target_ind7 = calc_target_index(state7, cx7, cy7)


    # 不断执行更新操作
    animation_interval=1 #每隔10个时间步更新一次动画
    animation_counter=0

    x=[state2.x]
    y=[state2.y]
    ai2=2
    x1h=[] ; y1h=[]
    # x11h = []; y11h = []
    x2h = [];y2h = []
    x3h=[] ; y3h=[]
    x4h = [];y4h = []
    x5h = [];y5h = []
    x6h = [];y6h = []
    x7h = [];y7h = []


    longitudinal=0
    d1=1 ;d2=1;d3=1 ;dz=1  #authority for driver
    sequence=[]
    while T1 >= time1 and lastIndex1 > target_ind1:
        time.sleep(0.1)
        ttime=ttime+0.1
        ai1 = PIDControl(target_speed1, state1.v)
        # ai11 = PIDControl(target_speed11, state11.v)
        if longitudinal==0:
            ai2 = PIDControl(target_speed2, state2.v)
        ai3 = PIDControl(target_speed3, state3.v)
        ai4 = PIDControl(target_speed4, state4.v)
        ai5 = PIDControl(target_speed5, state5.v)
        ai6 = PIDControl(target_speed6, state6.v)
        ai7 = PIDControl(target_speed7, state7.v)

        x1h.append(state1.x);y1h.append(state1.y)
        # x11h.append(state11.x);y11h.append(state11.y)
        x2h.append(state2.x);y2h.append(state2.y)
        x3h.append(state3.x);y3h.append(state3.y)
        x4h.append(state4.x);y4h.append(state4.y)
        x5h.append(state5.x);y5h.append(state5.y)
        x6h.append(state6.x);y6h.append(state6.y)
        x7h.append(state7.x);y7h.append(state7.y)


        di1, target_ind1,_ = pure_pursuit_control(state1, cx1, cy1, target_ind1)
        # di11, target_ind11, _ = pure_pursuit_control(state11, cx11, cy11, target_ind11)

        di2, target_ind2,rou_pp = pure_pursuit_control(state2, cx2, cy2, target_ind2)  #目标车辆
        # di3, target_ind3,_ = pure_pursuit_control(state3, cx3, cy3, target_ind3)
        di4, target_ind4,_ = pure_pursuit_control(state4, cx4, cy4, target_ind4)
        # di5, target_ind5, _ = pure_pursuit_control(state5, cx5, cy5, target_ind5)
        # di6, target_ind6, _ = pure_pursuit_control(state6, cx6, cy6, target_ind6)
        # di7, target_ind7, _ = pure_pursuit_control(state7, cx7, cy7, target_ind7)

        # zong he ren ji jieguo
        if state2.x > 130:
            di3 = di2 * dz + (1 - dz) * di4
            di6 = di2 * d2 + (1 - d2) * di4
            di7 = di2 * d3 + (1 - d3) * di4
            di5 = di2 * d1 + (1 - d1) * di4

            state3.yaw = state2.yaw * dz + (1 - dz) * state4.yaw

            state5.yaw=abs(state5.yaw)
            state6.yaw = abs(state6.yaw)
            state7.yaw = abs(state7.yaw)
            print(
                f"state2.yaw,state3.yaw,state4.yaw,state85.yaw,state6.yaw,state7.yaw:{state2.yaw, state3.yaw, state4.yaw, state5.yaw, state6.yaw, state7.yaw}")
            print(f"di1,di2,di3,di4,di85,di6,di7:{di1, di2, di3, di4, di5, di6, di7}")
        else:
            di3 = di2
            di4 = di2
            di6 = di2
            di7 = di2
            di5 = di2
            state3.yaw = state2.yaw
            state4.yaw = state2.yaw
            state5.yaw=state2.yaw
            state6.yaw = state2.yaw
            state7.yaw = state2.yaw



        state1 = update(state1, ai1, di1)
        # state11 = update(state11, ai11, di11)
        state2 = update(state2, ai2, di2)
        state3 = update(state3, ai3, di3)
        state4 = update(state4, ai4, di4)
        state5 = update(state5, ai5, di5)
        state6 = update(state6, ai6, di6)
        state7 = update(state7, ai7, di7)


        time1 = time1 + dt
        x1.append(state1.x);y1.append(state1.y);yaw1.append(state1.yaw);v1.append(state1.v);t1.append(time1)
        # x11.append(state11.x);y11.append(state11.y);yaw11.append(state11.yaw);v11.append(state11.v)
        x2.append(state2.x);y2.append(state2.y);yaw2.append(state2.yaw);v2.append(state2.v)
        x3.append(state3.x);y3.append(state3.y);yaw3.append(state3.yaw);v3.append(state3.v)
        x4.append(state4.x); y4.append(state4.y);yaw4.append(state4.yaw);v4.append(state4.v)
        x5.append(state5.x);y5.append(state5.y);yaw5.append(state5.yaw);v5.append(state5.v)
        x6.append(state6.x);y6.append(state6.y); yaw6.append(state6.yaw);v6.append(state6.v)
        x7.append(state7.x);y7.append(state7.y);yaw7.append(state7.yaw);v7.append(state7.v)

        yc1x = state1.x
        yc1y = state1.y
        yc2x = state2.x
        yc2y = state2.y
        yc3x = state3.x
        yc3y = state3.y
        yc4x = state4.x
        yc4y = state4.y


        yc1x = torch.tensor(yc1x)
        yc1y = torch.tensor(yc1y)
        yc2x = torch.tensor(yc2x)
        yc2y = torch.tensor(yc2y)
        yc3x = torch.tensor(yc3x)
        yc3y = torch.tensor(yc3y)
        yc4x = torch.tensor(yc4x)
        yc4y = torch.tensor(yc4y)

        z = torch.tensor(1)

        yc1_1 = torch.stack([yc1x, z]).unsqueeze(0)
        yc2_1 = torch.stack([yc2x, z]).unsqueeze(0)
        yc3_1 = torch.stack([yc3x, z]).unsqueeze(0)
        yc4_1 = torch.stack([yc4x, z]).unsqueeze(0)

        yc1_2 = torch.stack([yc1y, z]).unsqueeze(0)
        yc2_2 = torch.stack([yc2y, z]).unsqueeze(0)
        yc3_2 = torch.stack([yc3y, z]).unsqueeze(0)
        yc4_2 = torch.stack([yc4y, z]).unsqueeze(0)

        xy_target1_1 = torch.cat([xy_target1_1, yc1_1], dim=0)
        xy_target2_1 = torch.cat([xy_target2_1, yc2_1], dim=0)
        xy_target3_1 = torch.cat([xy_target3_1, yc3_1], dim=0)
        xy_target4_1 = torch.cat([xy_target4_1, yc4_1], dim=0)

        xy_target1_2 = torch.cat([xy_target1_2, yc1_2], dim=0)
        xy_target2_2 = torch.cat([xy_target2_2, yc2_2], dim=0)
        xy_target3_2 = torch.cat([xy_target3_2, yc3_2], dim=0)
        xy_target4_2 = torch.cat([xy_target4_2, yc4_2], dim=0)



        # 设计一条人驾驶的报警轨迹 1号车
        if 0<state2.x <10:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 5.75)
        if 10<=state2.x <20:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 4.75)
        if 20<=state2.x <30:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 5.75)
        if 30<=state2.x <40:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 4.75)
        if 40<=state2.x <50:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 5.75)
        if 50<=state2.x <60:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 4.75)
        if 60<=state2.x <70:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 5.25)

        # 设计一条人驾驶的报警轨迹 2号车
        if state2.x >140:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, 12 + 1.75)
        if state2.y > 8.25:
            cx2 = np.arange(state2.x, 280, 0.01)  # 目标车辆轨迹
            cy2 = np.full_like(cx2, state2.y)


        # 设计一条人驾驶的报警轨迹 4号车
        if state4.x > 140:
            cx4 = np.arange(state4.x, 280, 0.01)  # 目标车辆轨迹
            cy4 = np.full_like(cx4, 9 + 1.75)
        if state4.y > 5+1.75:
            cx4 = np.arange(state4.x, 280, 0.01)  # 目标车辆轨迹
            cy4 = np.full_like(cx4, state4.y)

        #calculate data for 4.2
        Vx = state3.v * math.cos(state3.yaw)
        Vy = state3.v * math.sin(state3.yaw)
        ax = ai3 * math.cos(state3.yaw)
        ay = ai3 * math.sin(state3.yaw)
        P = abs(state3.y - 5.25)
        # if state3.y<=7:
        #     P=abs(state3.y-5.25)
        # else:
        #     P = abs(state3.y - 8.75)

        data=[Vx,Vy,ax,ay,P,state3.yaw]
        if len(sequence)<=100:
            sequence.append(data)
        if len(sequence)>100:
            sequence.pop(0)
            sequence.append(data)



        if len(xy_target1_1) >= 31:
            if state2.x > 89:
                xy_target1_1 = torch.tensor(xy_target1_1)
                xy_target2_1 = torch.tensor(xy_target2_1)
                xy_target3_1 = torch.tensor(xy_target3_1)
                xy_target4_1 = torch.tensor(xy_target4_1)

                xy_target1_2 = torch.tensor(xy_target1_2)
                xy_target2_2 = torch.tensor(xy_target2_2)
                xy_target3_2 = torch.tensor(xy_target3_2)
                xy_target4_2 = torch.tensor(xy_target4_2)

                last_16_coords1_1 = xy_target1_1[-30:]
                last_16_coords2_1 = xy_target2_1[-30:]
                last_16_coords3_1 = xy_target3_1[-30:]
                last_16_coords4_1 = xy_target4_1[-30:]

                last_16_coords1_2 = xy_target1_2[-30:]
                last_16_coords2_2 = xy_target2_2[-30:]
                last_16_coords3_2 = xy_target3_2[-30:]
                last_16_coords4_2 = xy_target4_2[-30:]

                last_16_coords1_1 = last_16_coords1_1.unsqueeze(1)
                last_16_coords2_1 = last_16_coords2_1.unsqueeze(1)
                last_16_coords3_1 = last_16_coords3_1.unsqueeze(1)
                last_16_coords4_1 = last_16_coords4_1.unsqueeze(1)

                last_16_coords1_2 = last_16_coords1_2.unsqueeze(1)
                last_16_coords2_2 = last_16_coords2_2.unsqueeze(1)
                last_16_coords3_2 = last_16_coords3_2.unsqueeze(1)
                last_16_coords4_2 = last_16_coords4_2.unsqueeze(1)

                #
                #第2个点预测
                combined_coords2_1 = torch.tensor(last_16_coords2_1).cuda()
                combined_coords2_2 = torch.tensor(last_16_coords2_2).cuda()
                obs_traj_rel = torch.zeros((16, 1, 2)).cuda()
                relative_coords_1 = torch.diff(combined_coords2_1, dim=0)
                relative_coords_2 = torch.diff(combined_coords2_2, dim=0)
                zero_matrix_1 = torch.zeros(1, 1, 2).cuda()
                zero_matrix_2 = torch.zeros(1, 1, 2).cuda()
                obs_traj_rel1_1 = torch.cat((zero_matrix_1, relative_coords_1), dim=0)
                obs_traj_rel1_2 = torch.cat((zero_matrix_2, relative_coords_2), dim=0)
                obs_traj1_1 = combined_coords2_1
                obs_traj1_2 = combined_coords2_2
                seq_start_end = np.array([[0, 1]])
                seq_start_end = torch.tensor(seq_start_end).cuda()
                pred_traj_fake_1, _ = prediction_fake(obs_traj1_1, obs_traj_rel1_1, seq_start_end)
                pred_traj_fake_2, _ = prediction_fake(obs_traj1_2, obs_traj_rel1_2, seq_start_end)
                pred_traj_fake_1=pred_traj_fake_1*4
                pred_traj_fake_2 = pred_traj_fake_2*0.6
                xcod = pred_traj_fake_1[:, :, 0]-3*state2.x
                ycod = pred_traj_fake_2[:, :, 0]+0.4*state2.y
                pred_traj_fake2_1 = torch.stack((xcod, ycod), dim=2)  #预测的坐标点

               

                #第三辆车的预测点
                pred=pred_traj_fake2_1[-1]
                pred_x3_1 =pred[:,0]
                pred_y3_2 = pred[:, 1]
                #第四辆车的预测点
                # pred=pred_traj_fake4_1[-1]
                # pred_x4_1 =pred[:,0]
                # pred_y4_2 = pred[:, 1]


            U_apf=apf_force( state3.y,  state1.y)   #计算与1号车的风险

            #计算横向位置的风险(APF)
            # 测试计算ego和3号车之间的DPF风险
            delta_v=state1.v-state3.v
            delta_x=state1.x-state3.x
            delta_a=ai3-ai1
            V3 = state3.v
            U_dpf = dpf_velocity(delta_v, delta_x, delta_a,V3)

            # authority assignment
            #guiyihua
            if U_apf <= 0.3:
                U_apf1 = 0
            elif 0.3 < U_apf < 4.06:
                U_apf1 = (U_apf - 0.3) / (4.06 - 0.3)
            else:
                U_apf1 = 1

            if U_dpf <= 0.0001:
                U_dpf1 = 0
            elif 0.0001 < U_dpf < 150:
                U_dpf1 = (U_dpf - 0.0001) / (150 - 0.0001)
            else:
                U_dpf1 = 1

            # calculate longitudinal and lateral allocation (dynamic max point)
            b_max=0.4
            if len(yaw3)>=2:
                yy3=abs((yaw3[-1]-yaw3[-2])/0.1)
                if yy3<=0.4:
                    b=51*((yy3-0)/(0.4-0))
                else:
                    b=51*((0.4-0)/(0.4-0))

            if state2.x > 100:
            # 4.1 yitu yizhi xing
                hx=xy_target2_1[:,0]
                hy = xy_target2_2[:, 0]
                xy_2 = torch.stack((hx, hy),dim=1)
                xy_2 =xy_2[-10:]
                # print(xy_3)
                pred_2=pred_traj_fake2_1.cpu()
                pred_2 =torch.squeeze(pred_2,1)
                pred_2 = pred_2[:10]
                xy_21 = xy_2[:, 1]
                pred_21 = pred_2[:, 0]
                xy_2 = torch.stack((pred_21, xy_21), dim=1)
                wi = compute_rho_trajectory(xy_2, pred_2)
            else:
                wi = 0

        # 4.2 to evaluate driver ability
            if len(sequence) >= 100:
                wd=func_d(sequence)
            else:
                wd=1
            wd=wd*0.85
        # 4.3kongjian pengzhuang fengxian quanxian fenpei
            d11=U_apf1*U_dpf1
            ws=func(d11,b)

            #authority allocation
            d1=wi  ; d2=wd  ;  d3=ws
            dz=fz_func(wi,wd,ws)

        #printer
        print(f"state1.x,state1.y,state2.x,state2.y,state3.x,state3.y,state4.x,state4.y,state5.x,state5.y,state6.x,state6.y,state7.x,state7.y"
              f":{state1.x,state1.y,state2.x,state2.y,state3.x,state3.y,state4.x,state4.y,state5.x,state5.y,state6.x,state6.y,state7.x,state7.y}")
        print(f"d1,d2,d3,dz:{d1,d2,d3,dz}")

        if state2.x > 250:  #stop program
            break
        if state2.x > 89:
            # 指定保存的CSV文件路径
            file_path = "data_tensor.csv"
            # 检查文件是否存在，并以追加模式打开文件
            mode = 'a'  # 使用追加模式
            if not os.path.exists(file_path):
                mode = 'w'  # 如果文件不存在，使用写模式来创建文件并写入标题行
            # 打开文件进行写入
            with open(file_path, mode, newline='') as file:
                writer = csv.writer(file, delimiter='\t')  # 使用制表符作为分隔符
                # 如果文件是新创建的，写入标题行（如果需要）
                writer.writerow(["state1.x","state1.y","state2.x","state2.y","state3.x","state3.y","state4.x","state4.y","state5.x","state5.y","state6.x","state6.y","state7.x","state7.y"])
                vehicle_data = [state1.x,state1.y,state2.x,state2.y,state3.x,state3.y,state4.x,state4.y,state5.x,state5.y,state6.x,state6.y,state7.x,state7.y]   # 获取当前车辆的数据
                writer.writerow(vehicle_data)    # 将数据写入文件
                writer.writerow(["d1", "d2", "d3", "dz"])
                vehicle_data = [d1,d2,d3,dz]
                writer.writerow(vehicle_data)
                writer.writerow(["state2.yaw","state3.yaw", "state4.yaw", "state5.yaw", "state6.yaw", "state7.yaw"])
                vehicle_data = [state2.yaw, state3.yaw, state4.yaw, state5.yaw, state6.yaw, state7.yaw]
                writer.writerow(vehicle_data)
                writer.writerow(["di1", "di2", "di3", "di4", "di5", "di6", "di7"])
                vehicle_data = [di1, di2, di3, di4, di5, di6, di7]
                writer.writerow(vehicle_data)
                writer.writerow(["U_apf", "U_dpf"])
                vehicle_data = [U_apf, U_dpf]
                writer.writerow(vehicle_data)
                print(f"数据已按帧和车辆编号追加保存到 {file_path}")

        animation_counter+=1
        if animation_counter>=animation_interval:
            animation_counter=0
            if show_animation:
                plt.cla()
                plt.plot(bx1, by1, linewidth=1.5, color='k')
                plt.plot(bx2, by2, linewidth=1.5, color='k')
                #绘制车辆
                draw.draw_car(state1.x, state1.y, state1.yaw, di1, C)
                # draw.draw_car(state11.x, state11.y, state11.yaw, di11, C)
                draw.draw_car(state2.x, state2.y, state2.yaw, di2, C)
                draw.draw_car(state3.x, state3.y, state3.yaw, di3, C)
                draw.draw_car(state4.x, state4.y, state4.yaw, di4, C)
                draw.draw_car(state5.x, state5.y, state5.yaw, di5, C)
                draw.draw_car(state6.x, state6.y, state6.yaw, di6, C)
                draw.draw_car(state7.x, state7.y, state7.yaw, di7, C)

                plt.gcf().set_size_inches(20,6)   # 设置图的大小
                plt.subplots_adjust(left=0.04, right=1, bottom=0.04, top=0.97) # 设置图的左、右、底部和顶部的空白

                plt.plot(cx1, cy1, linestyle='--', color='gray', linewidth=2)
                plt.plot(state2.x, state2.y, linestyle='--', color='gray', linewidth=2)
                plt.plot(cx3, cy3, linestyle='--', color='gray', linewidth=2)
                plt.plot(cx4, cy4, linestyle='--', color='gray', linewidth=2)
                plt.plot(cx5, cy5, linestyle='--', color='gray', linewidth=2)
                plt.plot(cx6, cy6, linestyle='--', color='gray', linewidth=2)
                plt.plot(cx7, cy7, linestyle='--', color='gray', linewidth=2)

                x.append(state3.x)
                y.append(state3.y)
                plt.plot(x,y,"-b")

                if len(xy_target1_1) >= 31:

                    if state2.x > 89:
                        pred_traj_fake2_1 = pred_traj_fake2_1.cpu().detach().numpy()
                        plt.plot(pred_traj_fake2_1[:, :, 0], pred_traj_fake2_1[:, :, 1], 'ro-', markersize=5)
                    #
                plt.plot(x, y, "-b", label="trajectory")
                plt.plot(cx3, cy3, "-b", label="target")
                plt.axis("equal")
                plt.grid(True)
                plt.title("Speed[km/h]:" + str(state2.v * 3.6)[:4])
                plt.pause(0.01)
    plt.show()



if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
