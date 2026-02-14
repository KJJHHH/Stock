import yfinance as yf

symbol = "2330.TW"
df = yf.Ticker(symbol).history(start="2010-01-01", end="2024-12-31")

print(df.head())
print(f"rows={len(df)}")
