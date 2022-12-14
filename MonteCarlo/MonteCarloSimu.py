import streamlit as st
import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

class Simulation:

    @staticmethod
    def simulate(business_time = 14 * 60, scale1 = 1.35, scale2 = 1):
        """模拟一天的排队情况.1个DataFrame为1次模拟的结果"""
        guest = pd.DataFrame([[np.random.exponential(scale1), 0, np.random.exponential(scale2)]])
        while guest.iloc[0, 0] + guest.iloc[:, 2].sum() <= business_time:
            interval = np.random.exponential(scale1)
            new_guest = pd.DataFrame([[interval, max(0, guest.iloc[-1, 1] + guest.iloc[-1, 2] - interval), np.random.exponential(scale2)]])
            guest = pd.concat([guest, new_guest], ignore_index=True)
        
        guest.columns = ['到达时间间隔', '等待时间', '服务时间']
        return guest

    @staticmethod
    def monteCarlo(simulate, n_epoch = 100, scale1=1.35, scale2=1, log = None) -> list[pd.DataFrame]:
        """进行n次模拟——MonteCarlo"""
        result = []
        for i in range(n_epoch):
            result.append(simulate(scale1=scale1, scale2=scale2))
            try:
                if log != None:
                    log.progress((i + 1) / n_epoch)
            except:
                pass

        return result

    ####################################################################################################################################

    @staticmethod
    def analyze(result: list[pd.DataFrame]) -> dict[4]:
        """统计分析——回答前3问"""
        analysis = {'avg':[], 'max':[], 'over_2_min':0, 'total': 0}
        for guest in result:
            analysis['avg'].append(guest.iloc[:, 1].mean())
            analysis['max'].append(guest.iloc[:, 1].max())
            analysis['over_2_min'] += guest.iloc[:, 1][guest.iloc[:, 1] > 2].shape[0]
            analysis['total'] += guest.iloc[:, 0].shape[0]

        output = {
            'avg': np.mean(analysis['avg']), 
            'max': np.max(analysis['max']), 
            'over_2_min': analysis['over_2_min'],
            'total': analysis['total']}
        return output

    @staticmethod
    def visualize(result: list[pd.DataFrame]) -> plt.Figure:
        """绘制直方图"""
        waiting_time = result[0].iloc[:, 1]
        for new_guest in result[1: ]:
            waiting_time = pd.concat([waiting_time, new_guest.iloc[:, 1]], ignore_index=True)         

        fig, ax = plt.subplots(dpi=120)
        ax.hist(waiting_time, bins=100, density=True)
        ax.grid(linestyle='--', alpha=0.8)
        ax.set_title("Frequency Density Histogram")
        ax.set_xlabel("Waiting time")
        ax.set_ylabel("Density")
        return fig


def page_init():
    st.markdown("""
    # Burger Dome快餐店 模拟分析
    ## 问题描述
    - 店员服务完上一个顾客才能接待下一个顾客
    - 顾客到达的时间间隔 ~ Exp(1.35)
    - 店员服务时间 ~ Exp(1)
    """)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['问题1', '问题2', '问题3', '问题4', '问题5'])
    with tab1:
        st.write("每位顾客平均等待时间?")
    with tab2:
        st.write("最长等待时间?")
    with tab3:
        st.write("顾客等待时间超过2分钟的概率?")
    with tab4:
        st.write("直方图反映等待时间的分布?")
    with tab5:
        st.write("不断增加试验次数, 会减少汇总统计量的变异性。为什么对Burger Dome, 这种做法不合适?")

if __name__ == '__main__':

    page_init()
    st.markdown("## 模拟结果")

    scale1 = st.sidebar.slider("平均到达时间间隔", value=1.35, min_value=0.5, max_value=3.5, step=0.05)
    scale2 = st.sidebar.slider("平均服务时间", value=1., min_value=0.5, max_value=3.5, step=0.05)
    n_epoch = st.sidebar.slider("迭代次数", value=100, min_value=1, max_value=300)
    flag = st.sidebar.button("Run")

    if flag == True:

        with st.spinner("Loading..."):
            log = st.progress(0)
            result = Simulation.monteCarlo(Simulation.simulate, n_epoch, scale1=scale1, scale2=scale2, log=log)

        analysis = Simulation.analyze(result)
        st.write(f"平均等待时间: {analysis['avg']: .2f} min")
        st.write(f"最长等待时间: {analysis['max']: .2f} min")
        st.write(f"等待时间超过2min的概率: {analysis['over_2_min'] * 100 / analysis['total']: .2f}%")
        st.write(f" ( 平均接待顾客人数: {int(analysis['total'] / n_epoch)}人 ) ")
        
        with st.spinner("Drawing Picture..."):
            figure = Simulation.visualize(result)
            st.write(figure)
            st.balloons()
    