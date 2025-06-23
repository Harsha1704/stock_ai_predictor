import yfinance as yf
import plotly.graph_objs as go

def get_candlestick(symbol: str):
    df = yf.download(symbol, period="7d", interval="1h")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])
    fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_rangeslider_visible=False)
    return fig
