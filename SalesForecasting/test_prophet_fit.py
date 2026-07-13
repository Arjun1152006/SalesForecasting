import pandas as pd
from prophet import Prophet
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

df = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
    'y': list(range(10))
})

try:
    print("Initializing Prophet...")
    p = Prophet()
    print("Fitting Prophet...")
    p.fit(df)
    print("Prophet fit works!")
except Exception as e:
    print("Failed to fit Prophet:")
    import traceback
    traceback.print_exc()
