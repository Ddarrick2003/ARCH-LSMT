import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Financial Forecasting Dashboard", layout="wide")
st.title("📊 Financial Forecasting App (LSTM + GARCH)")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("🧾 Preview of uploaded data:", df.head())
        df['Returns'] = df['Returns'].fillna(0)
        df, _, _ = preprocess_data(df)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()

    st.success(f"✅ Cleaned data shape: {df.shape}")

    if df.shape[0] < 60:
        st.warning("⚠️ Very few rows. LSTM may not work well. GARCH may still run.")

    tab1, tab2 = st.tabs(["📈 LSTM Forecast", "📉 GARCH Risk Forecast"])

    with tab1:
        st.subheader("LSTM Forecasting")
        try:
            features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
            X, y = create_sequences(df[features], target_col='Close')
            if len(X) == 0:
                st.warning("⚠️ Not enough data for sequence modeling.")
            else:
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                model.fit(X_train, y_train, epochs=10, batch_size=16,
                          validation_data=(X_test, y_test),
                          callbacks=[EarlyStopping(patience=3)], verbose=0)
                preds = model.predict(X_test).flatten()
                st.line_chart({"Actual": y_test[:100], "Predicted": preds[:100]})
        except Exception as e:
            st.error(f"LSTM failed: {e}")

    with tab2:
        st.subheader("GARCH Risk Forecasting")
        try:
            vol_forecast, var_1d = forecast_garch_var(df)

            # Display metrics
            st.metric(label="1-Day VaR (95%)", value=f"{var_1d:.2f}%")
            st.line_chart(vol_forecast.values)

            # Explanation block
            st.markdown("### 🧠 Interpretation of Risk Forecast")
            st.info(f"""
            ✅ **Volatility Forecast (Chart)**:
            - The chart above shows expected volatility over the next {len(vol_forecast)} days.
            - Sharp spikes suggest periods of **increased uncertainty or market turbulence**.
            - Stable flat lines indicate a **more predictable market**.

            ✅ **Value at Risk (VaR)**:
            - The 1-Day Value at Risk (VaR) is estimated at **{abs(var_1d):.2f}%**.
            - This means: *With 95% confidence, the model expects the portfolio will not lose more than {abs(var_1d):.2f}% in a single day.*

            📌 **Use Case**: Traders, investors, and analysts use this to manage portfolio exposure and set stop-loss levels.
            """)

        except Exception as e:
            st.error(f"GARCH failed: {e}")

else:
    st.info("📥 Please upload a CSV to begin.")
