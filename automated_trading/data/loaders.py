import yfinance as yf
import pandas as pd
from typing import List
from .file_handler import LocalFileHandler

CACHE_FILE = 'finance_data.csv'
TIMESTAMP_FILE = 'last_update.txt'


class DataLoader:
    """
    数据加载与缓存管理，支持文件系统的依赖注入。
    """

    def __init__(self, symbols: List[str], cache_file: str = CACHE_FILE, timestamp_file: str = TIMESTAMP_FILE,
                 file_handler=None):
        self.symbols = symbols
        self.cache_file = cache_file
        self.timestamp_file = timestamp_file
        self.file_handler = file_handler or LocalFileHandler()  # 默认使用本地文件系统

    def is_data_outdated(self) -> bool:
        """
        检查缓存数据是否过期（一天更新一次）。
        """
        if not self.file_handler.exists(self.timestamp_file):
            return True

        last_update = self.file_handler.read(self.timestamp_file).strip()
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        return today != last_update

    def update_timestamp(self):
        """
        更新时间戳文件。
        """
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        self.file_handler.write(self.timestamp_file, today)

    def fetch_data(self) -> pd.DataFrame:
        """
        从缓存或网络加载数据。
        """
        if self.file_handler.exists(self.cache_file) and not self.is_data_outdated():
            print("Loading data from cache...")
            return pd.read_csv(self.cache_file, index_col=0, parse_dates=True, header=[0, 1])

        print("Downloading new data...")

        start_date = pd.Timestamp("2015-01-01").strftime('%Y-%m-%d')
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        symbols_str = " ".join(self.symbols)
        data = yf.download(symbols_str, start=start_date, end=end_date, group_by='tickers')
        data.to_csv(self.cache_file)
        self.update_timestamp()

        return data
