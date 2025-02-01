# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

import os
import traceback
from datetime import datetime

    # 导入你已有的分析逻辑
    # 注意修改 import 语句以匹配你的文件/类名
from SP500 import QuantAnalyzer, get_start_date

app = FastAPI()

    # ============ 定义请求和响应模型 (可自定义) ============

class AnalysisRequest(BaseModel):
        """前端请求体的字段示例"""
        symbol: str = "^GSPC"          # 股票代码, 如 AAPL, TSLA, 或 ^GSPC (标普500)
        start_date: Optional[str] = None  # 起始日期, 不传则用默认

class AnalysisResponse(BaseModel):
        """返回结果给前端的字段示例"""
        success: bool
        message: str
        trades: Optional[list] = None
        future_signals: Optional[list] = None

    # ============ 路由示例：进行分析并返回结果 ============

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_stock(req: AnalysisRequest):
        """
        参数通过 POST 传递 symbol(股票代码) 和 start_date(起始日期).
        进行量化分析, 并返回一些关键信息给调用方.
        """
        try:
            # 1. 创建分析器实例
            analyzer = QuantAnalyzer()

            # 2. 确定要使用的起始日期
            start = req.start_date or get_start_date()

            # 3. 下载数据
            analyzer.download_data(symbol=req.symbol, start=start)

            # 4. 计算指标
            analyzer.calculate_indicators()

            # 5. 训练模型
            analyzer.train_model()

            # 6. 分析历史交易
            trades = analyzer.analyze_trades()

            # 7. 预测未来(返回预测值数组)
            future_predictions = analyzer.predict_future()

            # 将预测信号转成可读字符串
            future_signals = []
            for i, p in enumerate(future_predictions, 1):
                action = "买入" if p == 1 else ("卖出" if p == -1 else "持有")
                future_signals.append(action)

            # 8. 返回结果给前端
            return AnalysisResponse(
                success=True,
                message="分析完成！",
                trades=trades,
                future_signals=future_signals
            )

        except Exception as e:
            traceback.print_exc()
            return AnalysisResponse(
                success=False,
                message=f"分析失败: {str(e)}"
            )


@app.get("/")
def root():
        """
        根路径测试
        """
        return {"message": "Welcome to Quant API, try POST /analyze"}


    # 你也可以加更多接口，比如返回可视化图表的链接，
    # 或把图表保存成图片并提供下载/展示等。