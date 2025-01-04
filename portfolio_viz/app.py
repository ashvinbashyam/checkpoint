import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Constants for column names
POSITION_DATE_COL = 'Date'  # Used in position data
TRADE_DATE_COL = 'Trade Date'  # Used in trade data
TICKER_COL = 'Ticker'  # Used in both
DESCRIPTION_COL = 'Description'  # Used in both
NMV_COL = "Market Value"  # Used in position data
PRICE_COL = "Market Price"  # Used in position data
COST_COL = "$ Average Cost"  # Used in position data
QUANTITY_TRADE_COL = "Notional Quantity"  # Used in trade data
QUANTITY_POS_COL = "Quantity"  # Used in position data
TXN_TYPE_COL = 'Txn Type'  # Used in trade data

def validate_excel_structure(excel_file):
    """Validate that the Excel file has the required sheets."""
    required_sheets = ["ITD Trade Blotter", "ITD History Portfolio"]
    actual_sheets = excel_file.sheet_names
    
    missing_sheets = [sheet for sheet in required_sheets if sheet not in actual_sheets]
    if missing_sheets:
        raise ValueError(f"Missing required sheets: {missing_sheets}")

def validate_sheet_columns(df, required_columns, sheet_name):
    """Validate that the DataFrame has all required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {sheet_name}: {missing_columns}")

def load_data(portfolio_history_content):
    """Load and process the consolidated Excel file into DataFrames."""
    # Load Excel data
    excel_file = pd.ExcelFile(portfolio_history_content)
    
    # Validate Excel structure
    validate_excel_structure(excel_file)
    
    # Define required columns
    trade_columns = [TRADE_DATE_COL, TICKER_COL, DESCRIPTION_COL, 
                    QUANTITY_TRADE_COL, TXN_TYPE_COL]
    position_columns = [POSITION_DATE_COL, TICKER_COL, DESCRIPTION_COL,
                       NMV_COL, PRICE_COL, COST_COL, QUANTITY_POS_COL]
    
    # Load Trade Blotter
    trade_df = pd.read_excel(
        excel_file,
        sheet_name="ITD Trade Blotter",
        keep_default_na=True
    )
    validate_sheet_columns(trade_df, trade_columns, "ITD Trade Blotter")
    trade_df[TRADE_DATE_COL] = pd.to_datetime(trade_df[TRADE_DATE_COL]).dt.normalize()
    
    # Load Position History
    position_df = pd.read_excel(
        excel_file,
        sheet_name="ITD History Portfolio",
        keep_default_na=True,
        na_values=['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                   '1.#IND', '1.#QNAN', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null']
    )
    validate_sheet_columns(position_df, position_columns, "ITD History Portfolio")
    position_df[POSITION_DATE_COL] = pd.to_datetime(position_df[POSITION_DATE_COL]).dt.normalize()
    
    return trade_df, position_df

def find_or_upload_portfolio_file():
    """Search for the portfolio history file or allow user to upload it."""
    # Define directories to search
    search_paths = [
        os.path.expanduser("~/Downloads"),
        r"Z:\Shared\Internal\Biotech"
    ]

    # File pattern to search for
    portfolio_pattern = "Checkpoint Daily Portfolio History*.xlsx"

    for path in search_paths:
        # Generate full path with pattern
        full_pattern = os.path.join(path, portfolio_pattern)
        portfolio_files = glob.glob(full_pattern)
        if portfolio_files:
            # Open the file and return its content as bytes
            with open(portfolio_files[0], "rb") as f:
                return f.read()  # Read file content into memory as bytes

    # If no files found, prompt user to upload
    st.warning("Required file not found in default directories!")
    uploaded_file = st.file_uploader("Please upload the Checkpoint Daily Portfolio History file", type=["xlsx"])
    
    if uploaded_file is not None:
        return uploaded_file.getvalue()  # Return the file content as bytes

    return None

def process_ticker_data(position_df, trade_df, ticker, dates):
    """Process data for a single ticker."""
    result = pd.DataFrame(index=dates)
    
    ticker_positions = position_df[position_df[TICKER_COL] == ticker]
    if not ticker_positions.empty:
        position_grouped = ticker_positions.groupby(POSITION_DATE_COL).first()
        result[f'{ticker}_Price'] = position_grouped[PRICE_COL]
        result[f'{ticker}_AvgCostBasis'] = position_grouped[COST_COL]
        result[f'{ticker}_SharesOwned'] = position_grouped[QUANTITY_POS_COL]
    else:
        result[f'{ticker}_Price'] = np.nan
        result[f'{ticker}_AvgCostBasis'] = np.nan
        result[f'{ticker}_SharesOwned'] = np.nan
    
    result[f'{ticker}_TradeQuantity'] = 0.0
    ticker_trades = trade_df[trade_df[TICKER_COL] == ticker].copy()
    if not ticker_trades.empty:
        ticker_trades['TradeSize'] = ticker_trades[QUANTITY_TRADE_COL]
        
        trade_series = pd.Series(0.0, index=dates)
        trade_grouped = ticker_trades.groupby(TRADE_DATE_COL)['TradeSize'].sum()
        common_dates = dates.intersection(trade_grouped.index)
        trade_series[common_dates] = trade_grouped[common_dates]
        
        result[f'{ticker}_TradeQuantity'] = trade_series
    
    return result

def compute_portfolio_metrics(trade_df, position_df):
    """Compute required portfolio metrics."""
    dates = pd.DatetimeIndex(np.sort(position_df[POSITION_DATE_COL].unique()))
    result = pd.DataFrame(index=dates)
    
    portfolio_value = position_df.groupby(POSITION_DATE_COL)[NMV_COL].sum()
    result['Portfolio_Value'] = portfolio_value
    
    tickers = [t for t in position_df[TICKER_COL].unique() if t != 'USD']
    all_ticker_data = []
    
    for ticker in tickers:
        ticker_data = process_ticker_data(position_df, trade_df, ticker, dates)
        all_ticker_data.append(ticker_data)
    
    if all_ticker_data:
        result = pd.concat([result] + all_ticker_data, axis=1)
    
    return result

def create_stock_chart(computedFields, ticker, date_range):
    """Create stock-specific chart with price, trades, and shares owned."""
    filtered_data = computedFields[date_range[0]:date_range[1]]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    price_data = filtered_data[f'{ticker}_Price']
    y_min = price_data.min()
    y_max = price_data.max()
    y_range = y_max - y_min
    
    y_margin = y_range * 0.1
    y_min -= y_margin
    y_max += y_margin
    y_min = 0
    
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=price_data,
            mode='lines',
            name='Stock Price',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data[f'{ticker}_AvgCostBasis'],
            mode='lines',
            name='Average Cost Basis',
            line=dict(color='grey')
        ),
        secondary_y=False
    )
    
    trade_data = filtered_data[f'{ticker}_TradeQuantity']
    if not trade_data.empty:
        buys = trade_data[trade_data > 0]
        sells = trade_data[trade_data < 0]
        
        max_trade_size = max(
            buys.max() if not buys.empty else 0,
            abs(sells.min()) if not sells.empty else 0
        )
        
        max_marker_size = 40
        # Calculate area scaling factor to make areas proportional
        # If we want the largest trade to have area π(max_marker_size/2)^2,
        # and we want area proportional to trade size,
        # then we need the radius (size/2) to be proportional to sqrt(trade_size)
        area_scale = max_marker_size / (2 * np.sqrt(max_trade_size)) if max_trade_size > 0 else 0
        
        if not buys.empty:
            # Calculate marker sizes where area is proportional to trade size
            buy_sizes = 2 * area_scale * np.sqrt(buys)
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=filtered_data.loc[buys.index, f'{ticker}_Price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        color='green',
                        size=buy_sizes,
                        opacity=0.4
                    ),
                    text=buys,  # Include buy sizes in hover information
                    hovertemplate='<b>Buy</b><br>Shares: %{text}<br>Price: %{y:.2f}<br>Date: %{x}<extra></extra>'
                ),
                secondary_y=False
            )
        
        if not sells.empty:
            # Calculate marker sizes where area is proportional to trade size
            sell_sizes = 2 * area_scale * np.sqrt(sells)
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=filtered_data.loc[sells.index, f'{ticker}_Price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        color='red',
                        size=sell_sizes,
                        opacity=0.4
                    ),
                    text=abs(sells),  # Include sell sizes in hover information
                    hovertemplate='<b>Sell</b><br>Shares: %{text}<br>Price: %{y:.2f}<br>Date: %{x}<extra></extra>'
                ),
                secondary_y=False
            )
    
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data[f'{ticker}_SharesOwned'],
            mode='lines',
            name='Shares Owned',
            line=dict(color='purple', dash='dot')
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'{ticker}',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis2_title='Shares Owned',
        hovermode='x unified',
        autosize=True,
        height=900
    )
    
    return fig

def create_portfolio_chart(computedFields, date_range):
    """Create portfolio value chart with date range selection."""
    filtered_data = computedFields[date_range[0]:date_range[1]]
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data['Portfolio_Value'],
            mode='lines',
            name='Portfolio Value'
        )
    )
    
    fig.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='x unified',
        autosize=True,
        height=900
    )
    
    return fig

def init_session_state():
    """Initialize session state variables if they don't exist."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.computed_fields = None
        st.session_state.min_date = None
        st.session_state.max_date = None
        st.session_state.tickers = None
        st.session_state.retry_counter = 0

def load_initial_data():
    """Load data and compute metrics from the consolidated file."""
    try:
        portfolio_history_content = find_or_upload_portfolio_file()
        
        if not portfolio_history_content:
            st.error("No portfolio file provided! Please upload a valid file to proceed.")
            return False
        
        with st.spinner("Loading data..."):
            try:
                trade_df, position_df = load_data(portfolio_history_content)
                st.session_state.computed_fields = compute_portfolio_metrics(trade_df, position_df)
                st.session_state.min_date = st.session_state.computed_fields.index.min()
                st.session_state.max_date = st.session_state.computed_fields.index.max()
                st.session_state.tickers = [t for t in position_df[TICKER_COL].unique() if t != 'USD']
                st.session_state.initialized = True
            except ValueError as e:
                st.error(f"Error loading file: {str(e)}")
                return False
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                return False
        
        return True
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False

def main():
    """Main application entry point."""
    st.set_page_config(page_title="Portfolio Visualization Tool", layout="wide")
    
    init_session_state()
    
    if not st.session_state.initialized:
        if not load_initial_data():
            return
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Portfolio 1", "Stock 1"])
    
    if page == "Portfolio 1":
        st.title("Portfolio Analysis")
        
        min_date = st.session_state.min_date.date()
        max_date = st.session_state.max_date.date()
        
        selected_dates = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="portfolio_date_slider"
        )
        
        start_date, end_date = selected_dates
        
        fig = create_portfolio_chart(st.session_state.computed_fields, (start_date, end_date))
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Stock 1
        st.title("Stock Analysis")
        
        # Check if we have tickers available
        if not st.session_state.tickers:
            st.error("No ticker data available. Please ensure the portfolio file is loaded correctly.")
            return
            
        # Get default ticker safely
        default_ticker = str(st.session_state.tickers[0]) if st.session_state.tickers else ""
        
        ticker = st.text_input("Enter Ticker", value=default_ticker)
        
        if not ticker:
            st.warning("Please enter a ticker symbol")
            return
            
        # Ensure ticker is a string and handle case-insensitive comparison
        try:
            ticker_upper = str(ticker).upper()
            available_tickers = [str(t).upper() for t in st.session_state.tickers if pd.notna(t)]
            
            if ticker_upper in available_tickers:
                # Get the original case-sensitive ticker
                original_idx = [str(t).upper() for t in st.session_state.tickers].index(ticker_upper)
                ticker = str(st.session_state.tickers[original_idx])
                
                min_date = st.session_state.min_date.date()
                max_date = st.session_state.max_date.date()
                
                selected_dates = st.slider(
                    "Select Date Range",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),
                    format="YYYY-MM-DD",
                    key="stock_date_slider"
                )
                
                start_date, end_date = selected_dates
                
                try:
                    fig = create_stock_chart(st.session_state.computed_fields, ticker, (start_date, end_date))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart for {ticker}: {str(e)}")
            else:
                st.warning(f"Invalid ticker. Available tickers: {', '.join(str(t) for t in st.session_state.tickers if pd.notna(t))}")
        except Exception as e:
            st.error(f"Error processing ticker {ticker}: {str(e)}")
                    
if __name__ == "__main__":
    main()
