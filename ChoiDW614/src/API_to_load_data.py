import FinanceDataReader as fdr
import pandas as pd


def read_data(file_name, code, exchange_region, start_date=None, end_date=None, data=None):
    root = './bin/data/' + file_name + '.csv'
    exchange_map = {
        'KRX': 'KRX', '한국 거래소': 'KRX', 'Seoul': 'KRX',
        'NASDAQ': 'NASDAQ', '나스닥': 'NASDAQ',
        'NYSE': 'NYSE', '뉴욕증권거래소': 'NYSE',
        'AMEX': 'AMEX', '미국증권거래소': 'AMEX',
        'Shanghai': 'SSE', '상해': 'SSE', '상하이': 'SSE', 'SSE': 'SSE',
        'Shenzhen': 'SZSE', '심천': 'SZSE', 'SZSE': 'SZSE',
        'Hong Kong': 'HKEX', '홍콩': 'HKEX', 'HKEX': 'HKEX',
        'Tokyo': 'TSE', '도쿄': 'TSE', 'TSE': 'TSE'
    }

    df = fdr.DataReader(symbol=code, start=start_date, end=end_date, exchange=exchange_map[exchange_region], data_source=data)

    if df is None:
        return False

    dataframe = pd.DataFrame(df)
    dataframe.to_csv(root, header=True, index=True)
    return True
