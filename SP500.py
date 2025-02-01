#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  

"""  
stock_analyzer.py  

演示如何将“未来 10 天的预测信号”添加到图表中并显示预测依据，并集成风险管理功能。  
并通过 mplfinance 绘制 K 线图。  
"""  

import pandas as pd  
import numpy as np  
import yfinance as yf  
import talib  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report  
import matplotlib.pyplot as plt  
from datetime import datetime, timedelta  

import matplotlib  
matplotlib.rcParams['font.sans-serif'] = ['Songti SC', 'Helvetica']  
matplotlib.rcParams['axes.unicode_minus'] = False  

#########################  
# 风险管理模块  
#########################  
class RiskManager:  
    def __init__(self, stop_loss=-3.0, take_profit=10.0, position_size=0.1):  
        """  
        Args:  
            stop_loss (float): 止损线（%），如-3.0表示亏损3%时止损  
            take_profit (float): 止盈线（%），如10.0表示盈利10%时止盈  
            position_size (float): 单笔交易的资金或仓位比例  
        """  
        self.stop_loss = stop_loss  
        self.take_profit = take_profit  
        self.position_size = position_size  

    def check_risk(self, current_price, entry_price):  
        """  
        检查当前持仓盈亏比例是否触发风控  
        返回:  
            None: 未触发风控  
            'stop_loss': 触发止损退出  
            'take_profit': 触发止盈退出  
        """  
        if entry_price == 0:  
            return None  

        current_return = (current_price - entry_price) / entry_price * 100  

        if current_return <= self.stop_loss:  
            return 'stop_loss'  
        elif current_return >= self.take_profit:  
            return 'take_profit'  
        return None  

#########################  
# 量化分析主体  
#########################  
def get_start_date():  
    """  
    返回要使用的起始日期，可以在这里随时修改或从其他配置中读取。  
    """  
    return "2020-01-01"  

class QuantAnalyzer:  
    def __init__(self):  
        """  
        初始化分析器  
        """  
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)  
        self.df = None               # 原始或带指标的数据  
        self.prediction_days = 10    # 用于未来预测的天数  
        self.future_df = pd.DataFrame()  # 存储未来预测结果(含日期、信号、理由等)，方便绘图用  

        # 集成风险管理  
        self.risk_manager = RiskManager(stop_loss=-3.0, take_profit=10.0, position_size=0.1)  

        # 存储交易信息，供后续统计和可视化  
        self.trades = []  

    def download_data(self, symbol="^GSPC", start="2024-01-01"):  
        """  
        下载历史数据  
        Args:  
            symbol (str): 股票或指数代码 (默认标普500: ^GSPC)  
            start (str): 起始日期 YYYY-MM-DD  
        """  
        print("正在下载市场数据...")  
        self.df = yf.download(symbol, start=start, end=datetime.now().strftime('%Y-%m-%d'))  

        if self.df is None or len(self.df) == 0:  
            raise ValueError(f"无法下载到 {symbol} 的有效数据，请检查代码或更换股票代码。")  

        # 如果是多层列索引（有时 yfinance 会返回多层列），先扁平化  
        if isinstance(self.df.columns, pd.MultiIndex):  
            self.df.columns = [  
                "_".join([str(c).strip() for c in col if c])  
                for col in self.df.columns.to_flat_index()  
            ]  
        # 自动重命名，例如将 "Close_^GSPC" 重命名为 "Close"  
        self.df.rename(  
            columns=lambda c: c.replace("_"+symbol, ""),  
            inplace=True  
        )  

        print(f"下载完成，数据范围：{self.df.index[0]} 至 {self.df.index[-1]}")  
        print(f"数据共 {len(self.df)} 条记录，列名：{list(self.df.columns)}")  

    def calculate_indicators(self):  
        """  
        计算技术指标  
        """  
        print("\n计算技术指标...")  

        if self.df is None or len(self.df) < 100:  
            raise ValueError("数据量不足(少于 100 条)，无法进行指标计算。")  

        df = self.df.copy()  

        needed_cols = ["Open", "High", "Low", "Close", "Volume"]  
        for col in needed_cols:  
            if col not in df.columns:  
                raise ValueError(  
                    f"缺少列: {col}，请检查下载数据的列名。当前列：{list(df.columns)}"  
                )  

        df.dropna(subset=["Close", "High", "Low", "Volume"], inplace=True)  

        close_array = df["Close"].astype(np.float64).values.flatten()  
        high_array = df["High"].astype(np.float64).values.flatten()  
        low_array = df["Low"].astype(np.float64).values.flatten()  

        print(f"数据条数: {len(df)}")  
        print(f"Close 数组形状: {close_array.shape}, dtype: {close_array.dtype}")  

        try:  
            # 均线  
            df["MA5"] = df["Close"].rolling(window=5).mean()  
            df["MA10"] = df["Close"].rolling(window=10).mean()  
            df["MA20"] = df["Close"].rolling(window=20).mean()  
            df["MA30"] = df["Close"].rolling(window=30).mean()  
            df["MA60"] = df["Close"].rolling(window=60).mean()  

            # MACD  
            macd, signal, hist = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)  
            df["MACD"] = macd  
            df["Signal_Line"] = signal  
            df["MACD_Hist"] = hist  

            # RSI  
            df["RSI"] = talib.RSI(close_array, timeperiod=14)  

            # 动量  
            df["MOM"] = talib.MOM(close_array, timeperiod=10)  

            # 布林带  
            upper, middle, lower = talib.BBANDS(close_array, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)  
            df["BB_upper"] = upper  
            df["BB_middle"] = middle  
            df["BB_lower"] = lower  

            # 成交量指标  
            df["Volume_MA"] = df["Volume"].rolling(window=20).mean()  
            df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]  

            # ATR 波动性指标  
            df["ATR"] = talib.ATR(high_array, low_array, close_array, timeperiod=14)  

            # 价格变动百分比  
            df["Price_Change"] = df["Close"].pct_change()  

        except Exception as e:  
            import traceback  
            print("计算指标时发生错误：")  
            traceback.print_exc()  
            raise e  

        df.dropna(inplace=True)  
        self.df = df  

        print(f"技术指标计算完成，最终数据量：{len(self.df)} 条。")  

    def create_features(self):  
        """  
        构造特征(X) 和 标签(y)，标签列名 Signal  
        """  
        df = self.df.copy()  

        # 建立交易信号列  
        df["Signal"] = 0  

        # 简易买入条件(仅供示例)  
        buy_conditions = (  
            (df["MA5"] > df["MA20"]) &  
            (df["RSI"] < 70) &  
            (df["MACD_Hist"] > 0) &  
            (df["Close"] > df["BB_middle"]) &  
            (df["Volume_Ratio"] > 1)  
        )  

        # 简易卖出条件(仅供示例)  
        sell_conditions = (  
            (df["MA5"] < df["MA20"]) |  
            (df["RSI"] > 80) |  
            (df["MACD_Hist"] < 0) |  
            (df["Close"] < df["BB_middle"]) |  
            (df["Volume_Ratio"] > 2)  
        )  

        df.loc[buy_conditions, "Signal"] = 1  
        df.loc[sell_conditions, "Signal"] = -1  

        features = [  
            "MA5","MA10","MA20","MA30","MA60",  
            "MACD","Signal_Line","MACD_Hist",  
            "RSI","MOM","BB_upper","BB_middle","BB_lower",  
            "Volume_Ratio","ATR","Price_Change"  
        ]  

        X = df[features]  
        y = df["Signal"]  

        self.df = df  
        return X, y  

    def train_model(self):  
        """  
        训练模型并输出评估报告  
        """  
        print("\n训练模型...")  
        X, y = self.create_features()  

        if len(X) < 30:  
            raise ValueError("有效数据太少，无法训练模型，请检查是否成功计算指标。")  

        X_train, X_test, y_train, y_test = train_test_split(  
            X, y, test_size=0.2, random_state=42  
        )  
        self.model.fit(X_train, y_train)  
        y_pred = self.model.predict(X_test)  

        print("\n模型评估报告:")  
        print(classification_report(y_test, y_pred))  

    def analyze_trades(self):  
        """  
        分析历史交易，集成风险管理  
        """  
        print("\n分析历史交易(含风险管理)...")  
        df = self.df.copy()  
        trades = []  
        position = False  
        entry_price = 0.0  
        entry_date = None  

        for date, row in df.iterrows():  
            signal = row["Signal"]  

            # 如果当前持仓，先检查风控是否触发  
            if position:  
                # 检查止损/止盈  
                risk_status = self.risk_manager.check_risk(row["Close"], entry_price)  
                if risk_status is not None:  
                    exit_price = row["Close"]  
                    profit_pct = (exit_price - entry_price) / entry_price * 100  
                    trades.append({  
                        "entry_date": entry_date,  
                        "exit_date": date,  
                        "entry_price": entry_price,  
                        "exit_price": exit_price,  
                        "profit_pct": profit_pct,  
                        "holding_days": (date - entry_date).days,  
                        "exit_reason": risk_status  
                    })  
                    position = False  
                    continue  # 风控优先，触发后直接跳到下一天  

            # 买入逻辑  
            if signal == 1 and not position:  
                entry_price = row["Close"]  
                entry_date = date  
                position = True  

            # 卖出逻辑  
            elif signal == -1 and position:  
                exit_price = row["Close"]  
                profit_pct = (exit_price - entry_price) / entry_price * 100  
                trades.append({  
                    "entry_date": entry_date,  
                    "exit_date": date,  
                    "entry_price": entry_price,  
                    "exit_price": exit_price,  
                    "profit_pct": profit_pct,  
                    "holding_days": (date - entry_date).days,  
                    "exit_reason": "signal"  
                })  
                position = False  

        self.trades = trades  
        return trades  

    def predict_future(self):  
        """  
        使用倒数 prediction_days 条数据预测未来走势，并生成带日期的 future_df。  
        在这里也可以生成'预测理由'。  
        """  
        print("\n预测未来走势...")  
        df = self.df.copy()  
        if len(df) < self.prediction_days:  
            raise ValueError("可用于预测的数据不足。")  

        # 取最后若干行用来做特征  
        last_part = df.iloc[-self.prediction_days:].copy()  

        feature_cols, _ = self.create_features()  
        col_names = feature_cols.columns  

        # 进行预测  
        predictions = self.model.predict(last_part[col_names])  

        # 这里为了演示，我们构造未来日期索引（从数据最后一天之后开始）  
        last_date = df.index[-1]  
        future_dates = [last_date + timedelta(days=i) for i in range(1, self.prediction_days+1)]  

        # 构造一个 DataFrame 存放预测信号  
        pred_df = pd.DataFrame({  
            "Predicted_Signal": predictions  
        }, index=future_dates)  

        # 为了示例，这里做一个“预测理由”的简单示例  
        reasons = []  
        for sig in predictions:  
            if sig == 1:  
                reason_str = "买入理由: MA5>MA20 & RSI<70 等"  
            elif sig == -1:  
                reason_str = "卖出理由: MA5<MA20 或 RSI>80 等"  
            else:  
                reason_str = "持有理由: 条件均不满足"  
            reasons.append(reason_str)  
        pred_df["Reason"] = reasons  

        self.future_df = pred_df  
        return predictions  

    def plot_analysis(self):  
        """  
        可视化图表  
        这里额外把 future_df 的预测信号添加到图表中，并用文字注释说明理由。  
        """  
        if self.df is None or len(self.df) == 0:  
            print("没有可视化的数据，请先下载并计算指标。")  
            return  

        plt.figure(figsize=(15, 10))  

        # (1) 价格和均线  
        plt.subplot(2, 1, 1)  
        plt.plot(self.df.index, self.df["Close"], label="价格", alpha=0.7)  
        plt.plot(self.df.index, self.df["MA20"], label="20日均线", alpha=0.7)  
        plt.plot(self.df.index, self.df["MA60"], label="60日均线", alpha=0.7)  

        # 历史买入/卖出信号标记  
        buy_signals = self.df[self.df["Signal"] == 1]  
        sell_signals = self.df[self.df["Signal"] == -1]  
        plt.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="green", s=100, label="买入")  
        plt.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="red", s=100, label="卖出")  

        # 如果已经生成了 future_df，则在图中标注  
        if not self.future_df.empty:  
            # 这里没有真实的未来 Close 价格，可以先假设与最后一天价格相同，用来演示标记位置  
            last_close = self.df["Close"].iloc[-1]  
            future_buys = self.future_df[self.future_df["Predicted_Signal"] == 1]  
            future_sells = self.future_df[self.future_df["Predicted_Signal"] == -1]  
            future_holds = self.future_df[self.future_df["Predicted_Signal"] == 0]  

            # 使用散点将它们标在同一水平线上，并在时间轴延后  
            plt.scatter(future_buys.index, [last_close]*len(future_buys), marker="^", color="blue", s=100, label="未来买入预测")  
            plt.scatter(future_sells.index, [last_close]*len(future_sells), marker="v", color="orange", s=100, label="未来卖出预测")  
            plt.scatter(future_holds.index, [last_close]*len(future_holds), marker="o", color="gray", s=80, label="未来持有预测")  

            # 用文字注释标出“预测理由”  
            for idx, row in self.future_df.iterrows():  
                x_coord = idx  
                y_coord = last_close  
                plt.text(  
                    x_coord, y_coord,  
                    f"{row['Reason']}",  
                    fontsize=9, rotation=30, ha="left", va="bottom",  
                    color="black", alpha=0.8  
                )  

        plt.title("价格 & 交易信号 (包含未来预测)")  
        plt.xlabel("日期")  
        plt.ylabel("价格")  
        plt.legend()  
        plt.grid(True)  

        # (2) RSI & MACD  
        plt.subplot(2, 1, 2)  
        plt.plot(self.df.index, self.df["RSI"], label="RSI", color="purple")  
        plt.plot(self.df.index, self.df["MACD"], label="MACD", color="blue")  
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)  
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)  

        plt.title("RSI & MACD")  
        plt.xlabel("日期")  
        plt.ylabel("指标数值")  
        plt.legend()  
        plt.grid(True)  

        plt.tight_layout()  
        plt.show()  

        # 如果需要可在此添加风险收益曲线的可视化，如下示例（可取消注释使用）：  
        # self.plot_risk_analysis()  

    def plot_risk_analysis(self):  
        """  
        (可选) 风险收益分析图表示例  
        """  
        if not self.trades:  
            print("没有交易数据，无法绘制风险收益分析。")  
            return  

        df_trades = pd.DataFrame(self.trades)  
        df_trades = df_trades.dropna(subset=["exit_date", "profit_pct"])  
        if df_trades.empty:  
            print("交易数据为空，无法绘制风险收益分析。")  
            return  

        # 按照离场时间排序  
        df_trades.sort_values("exit_date", inplace=True)  

        returns = df_trades["profit_pct"].values  
        cumulative_returns = np.cumsum(returns)  
        dates = df_trades["exit_date"].values  

        plt.figure(figsize=(10, 5))  
        plt.plot(dates, cumulative_returns, label="累计收益(%)", color="green")  

        # 计算最大回撤  
        peak = np.maximum.accumulate(cumulative_returns)  
        drawdown = (cumulative_returns - peak) / peak * 100  
        plt.fill_between(dates, cumulative_returns, peak, where=(peak>cumulative_returns),  
                         color='red', alpha=0.3, label="回撤区间")  

        plt.title("风险收益分析")  
        plt.xlabel("交易结束日期")  
        plt.ylabel("收益 (%)")  
        plt.legend()  
        plt.grid(True)  
        plt.show()  

    def plot_candlestick(self):  
        """  
        使用 mplfinance 绘制 K 线图 (含均线和成交量)，可与已有分析图表配合使用。  
        """  
        import mplfinance as mpf  

        if self.df is None or len(self.df) == 0:  
            print("没有可视化的数据，请先下载并计算指标。")  
            return  

        # mplfinance 需要的 DataFrame: index=日期, 列=["Open","High","Low","Close","Volume",...]  
        df_candle = self.df.copy()  
        df_candle.index.name = 'Date'  

        # 绘制 K 线 + 移动均线 + 成交量  
        mpf.plot(  
            df_candle,  
            type='candle',         # 蜡烛图  
            mav=(20, 60),          # 叠加均线  
            volume=True,           # 底部显示成交量  
            figratio=(15, 10),     # 图表大小  
            title='K线图 (含20日/60日均线)',  
            style='yahoo'          # 主题风格  
        )  


def main():  
    try:  
        analyzer = QuantAnalyzer()  

        print("开始分析流程...")  

        # 1. 获取起始日期并下载数据  
        start_date = get_start_date()  
        analyzer.download_data(symbol="^GSPC", start=start_date)  

        # 2. 计算技术指标  
        analyzer.calculate_indicators()  

        # 3. 训练模型  
        analyzer.train_model()  

        # 4. 分析历史交易，含风险管理  
        trades = analyzer.analyze_trades()  
        if trades:  
            print("\n=== 历史交易统计 ===")  
            total_trades = len(trades)  
            print(f"总交易次数: {total_trades}")  
            profitable_trades = sum(1 for t in trades if t.get("profit_pct", 0) > 0)  
            print(f"盈利交易次数: {profitable_trades}")  
            print(f"亏损交易次数: {total_trades - profitable_trades}")  
            if total_trades > 0:  
                print(f"胜率: {profitable_trades / total_trades * 100:.2f}%")  

            returns = [t["profit_pct"] for t in trades if "profit_pct" in t]  
            if returns:  
                print(f"平均收益率: {np.mean(returns):.2f}%")  
                print(f"最大单笔收益: {max(returns):.2f}%")  
                print(f"最大单笔亏损: {min(returns):.2f}%")  

            ############################  
            # 风险指标: 夏普比率/最大回撤  
            ############################  
            if len(returns) > 1:  
                # 夏普比率 (简易版本，未考虑无风险利率 & 年化)  
                sharpe = np.mean(returns)/np.std(returns) * np.sqrt(252)  
                print(f"夏普比率(近似): {sharpe:.2f}")  

                # 最大回撤(基于交易序列的累计收益)  
                cumulative = np.cumsum(returns)  
                peak = np.maximum.accumulate(cumulative)  
                drawdown = (cumulative - peak) / peak * 100  
                max_drawdown = np.min(drawdown)  
                print(f"最大回撤: {max_drawdown:.2f}%")  

            # 止损次数  
            stop_loss_count = sum(1 for t in trades if t.get("exit_reason") == "stop_loss")  
            print(f"触发止损次数: {stop_loss_count}")  

        # 5. 预测未来(并同时生成带日期和理由的 future_df)  
        future_predictions = analyzer.predict_future()  
        print(f"\n未来 {analyzer.prediction_days} 天的预测信号:")  
        for i, p in enumerate(future_predictions, 1):  
            action = "买入" if p == 1 else ("卖出" if p == -1 else "持有")  
            print(f"第 {i} 天: {action}")  

        # 6. 可视化 (包含未来预测散点标记)  
        analyzer.plot_analysis()  

        # 如果想查看风险收益分析曲线，可执行:  
        # analyzer.plot_risk_analysis()  

        # 7. (可选) 绘制 K 线图  
        print("\n单独展示K线图:")  
        analyzer.plot_candlestick()  

    except Exception as e:  
        print("\n程序执行出错!")  
        print(str(e))  

if __name__ == "__main__":  
    main()