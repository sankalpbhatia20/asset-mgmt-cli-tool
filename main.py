from ml_models.lstm import lstm_model
import typer

app = typer.Typer()

# Defining a global variable to remember the stock name entered bu the user
global_stock = None

@app.command()
def stock(stock_ticker: str):
    """
    Enter the stock ticker you would like to start analysing

    Args:
    stock_ticker (str): The Stock Ticker/Symbol [Only available for Indian and US stocks]

    Example usage:
    $ python3 main.py stock TSLA
    
    """

@app.command()
def ml_stock_analysis(stock_ticker: str, model: str, time_period: int):
    """
    Analyze a stock using machine learning models or fundamental analysis.

    Args:
    stock_ticker (str): The stock symbol to analyze.
    model (str): The ML model to use (LSTM, SVM, Logistic Regression, etc).
    time_period (int): Time period for the data to be analysed in WEEKS

    Example usage:
    $ python main.py ml_stock_analysis TSLA LSTM 52 
    """
    # Your analysis logic goes here

    if model == "LSTM" or model == "lstm":
        lstm_model(stock_ticker)

if __name__ == "__main__":
    app()