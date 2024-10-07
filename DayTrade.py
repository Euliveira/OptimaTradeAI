import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import requests
import time

# Função para obter dados da Binance
def fetch_binance_data(symbol):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=1000'  # Usando 1h para prever em 4h

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200 or 'code' in data:
        raise Exception(f"Erro ao buscar dados da Binance: {data}")

    prices = [float(candle[4]) for candle in data]  # Fechamento do candle
    return prices

# Função para preparar os dados para o modelo LSTM
def prepare_data(prices, seq_len):
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))
    sequences = np.array([scaled_prices[i:i + seq_len] for i in range(len(scaled_prices) - seq_len)])
    return torch.tensor(sequences, dtype=torch.float32), scaler

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hx=None, cx=None):
        out, (hn, cn) = self.lstm(x, (hx, cx))
        out = self.fc(out[:, -1, :])
        return out, (hn, cn)

# Função de treinamento do modelo
def train_model(model, data, criterion, optimizer, num_epochs=10):
    model.train()
    batch_size = data.size(0)
    seq_len = data.size(1)
    hidden_size = model.hidden_size
    num_layers = model.num_layers
    
    hx = torch.zeros(num_layers, batch_size, hidden_size)
    cx = torch.zeros(num_layers, batch_size, hidden_size)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs, (hn, cn) = model(data, hx, cx)
        target = data[:, -1, :]
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Função de previsão
def predict(model, data, scaler):
    model.eval()
    batch_size = data.size(0)
    seq_len = data.size(1)
    hidden_size = model.hidden_size
    num_layers = model.num_layers
    
    hx = torch.zeros(num_layers, batch_size, hidden_size)
    cx = torch.zeros(num_layers, batch_size, hidden_size)

    with torch.no_grad():
        outputs, _ = model(data, hx, cx)
        
    predicted_price = scaler.inverse_transform(outputs.numpy())
    return predicted_price[-1, 0]

# Função principal
def main():
    symbols = ["BTCUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "SOLUSDT"]
    input_size = 1
    hidden_size = 50
    num_layers = 2
    num_epochs = 20
    learning_rate = 0.001
    seq_len = 10

    while True:
        for symbol in symbols:
            print(f'Treinando o modelo para {symbol}...')
            
            prices = fetch_binance_data(symbol)
            data, scaler = prepare_data(prices, seq_len)
            
            model = LSTMModel(input_size, hidden_size, num_layers)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            train_model(model, data, criterion, optimizer, num_epochs=num_epochs)
            
            print(f'\nFazendo previsões para {symbol}...')
            prices = fetch_binance_data(symbol)
            data, _ = prepare_data(prices, seq_len)
            prediction = predict(model, data, scaler)
            
            atual_preco = prices[-1]
            variacao_percentual = ((prediction - atual_preco) / atual_preco) * 100
            
            print(f'Preço Atual ({symbol}): {atual_preco:.2f}')
            print(f'Preço Previsto para 4 horas ({symbol}): {prediction:.2f}')  # Alterado para 4 horas
            print(f'Variação Percentual Prevista ({symbol}): {variacao_percentual:.2f}%')
            
            # Alertas sem Stop Loss
            if variacao_percentual > 0:
                print(f'Alerta de Compra para {symbol}: Previsão de alta no preço.')
            else:
                print(f'Alerta de Venda para {symbol}: Previsão de queda no preço.')
                
        print("\nAguardando 4 horas antes de repetir o processo...\n")  # Alterado para 4 horas
        time.sleep(14400)  # 4 horas = 14400 segundos

if __name__ == "__main__":
    main()
