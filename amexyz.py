import os
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, Counter
import random
import re
from datetime import datetime

# --- Configuration Parameters ---
# WINDOW_SIZE is still relevant for the sequence length passed to the Transformer
WINDOW_SIZE = 90
INITIAL_CASH = 10000
TRANSACTION_FEE_PERCENT = 0.001
CHECKPOINT_DIR = "amexyz/checkpoints_"+str(WINDOW_SIZE)
CHECKPOINT_INTERVAL = 10 # Save every 10 episodes for demonstration, you can change to 100
num_episodes = 4000

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
batch_size = 64

print("Current working directory:", os.getcwd())

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    print(f"📂 Created checkpoint directory: {CHECKPOINT_DIR}")
else:
    print(f"📁 Checkpoint directory already exists: {CHECKPOINT_DIR}")


# 1. Data Acquisition
def get_price_data(filename="btc_prices.csv"):
    """
    Downloads or loads Bitcoin (BTC-USD) historical 'Close' price data.
    If the file exists, it loads from there; otherwise, it downloads from Yahoo Finance.
    """
    if os.path.exists(filename):
        print(f"📁 Loading BTC data from {filename}")
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True, skiprows=[1, 2])
            if 'Close' not in df.columns:
                print("Warning: 'Close' column not found after skipping rows. Trying without skipping.")
                df = pd.read_csv(filename, index_col=0, parse_dates=True)

        except Exception as e:
            print(f"Error loading {filename} with skiprows: {e}. Trying without skipping rows.")
            df = pd.read_csv(filename, index_col=0, parse_dates=True)

        if 'Close' not in df.columns:
            raise ValueError(f"Error: 'Close' column not found in {filename}. Please check the CSV file.")

        df = df[['Close']]
        df = df.dropna()
        print(f"Loaded data shape: {df.shape}")

    else:
        print("🌐 Downloading BTC data from Yahoo Finance...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download("BTC-USD", start="2018-01-01", end=end_date)
        df.to_csv(filename)
        print(f"✅ Data saved to {filename}")

    return df['Close'].values.astype(np.float32)

# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is (seq_len, batch_size, feature_dim)
        return x + self.pe[:x.size(0), :]

# 2. Define Transformer-based DQN Model
class TransformerDQN(nn.Module):
    """
    Transformer-based Deep Q-Network (DQN) model for action prediction.
    Input: state (sequence of past prices + current normalized cash + current normalized crypto)
    Output: Q-values for each action (Hold, Buy 25%, ..., Sell 100%)
    """
    def __init__(self, feature_dim, nhead, num_encoder_layers, dim_feedforward, output_dim=9):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None # Not used in this setup
        self.pos_encoder = PositionalEncoding(feature_dim)
        encoder_layers = nn.TransformerEncoderLayer(feature_dim, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # The decoder needs to take the combined output of the transformer and the auxiliary features
        # The transformer outputs `feature_dim` (which is 1 in your current setup for price)
        # The auxiliary features are 2 (cash, crypto)
        # So, the input to the decoder will be feature_dim + 2
        self.decoder = nn.Linear(feature_dim + 2, output_dim) # Corrected input dimension for decoder
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # --- FIX IS HERE: Add aux_features to the forward method signature ---
    def forward(self, src, aux_features):
        # src expected shape: (batch_size, sequence_length, feature_dim)
        # aux_features expected shape: (batch_size, aux_feature_dim)
        src = self.pos_encoder(src) # Add positional encoding
        output = self.transformer_encoder(src)
        # For RL, typically we might take the output of the last token, or pool.
        # Here, we'll take the output of the last token in the sequence (corresponding to current info).
        last_token_output = output[:, -1, :] # (batch_size, feature_dim)

        # Concatenate the transformer's output with the auxiliary features
        combined_features = torch.cat((last_token_output, aux_features), dim=1) # dim=1 to concatenate horizontally

        output = self.decoder(combined_features) # Map to Q-values
        return output

# 3. Environment Simulator
class TradingEnv:
    """
    Simulates a simplified crypto trading environment with transaction costs.
    Actions: 0 (Hold), 1-4 (Buy percentages), 5-8 (Sell percentages)
    """
    def __init__(self, prices, window_size=WINDOW_SIZE, initial_cash=INITIAL_CASH, fee_percent=TRANSACTION_FEE_PERCENT):
        if len(prices) < window_size + 1:
            raise ValueError("Prices array is too short for the given window_size.")
        self.prices = prices
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.fee_percent = fee_percent

        self.min_price_overall = np.min(self.prices)
        self.max_price_overall = np.max(self.prices)

        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.current_step = self.window_size # Start from window_size to have enough history
        self.cash = self.initial_cash
        self.crypto = 0
        self.portfolio_value_history = [self.initial_cash]
        self.action_history = []
        self.price_history = []
        return self._get_state()

    def _get_state(self):
        """
        Returns the current state: a window of historical prices, with current cash and crypto
        integrated at each step of the sequence (or appended as separate features).
        For simplicity, let's include normalized cash and crypto as features at the *end*
        of the price sequence for the Transformer to process globally, or concatenate them.
        A more sophisticated approach might embed them into the sequence.

        For a Transformer, we want a sequence. Let's make each element in the sequence
        a tuple/vector of (price, normalized_cash_at_time_t, normalized_crypto_at_time_t).
        However, cash/crypto only change based on actions.
        A common approach for time series with external features:
        1. Sequence of prices.
        2. Append `[normalized_cash, normalized_crypto]` *once* at the end of the sequence,
            or provide them as separate inputs to the model.

        Let's try: sequence of `window_size` prices, and then append the current
        normalized cash and crypto values as part of the *last* element of the sequence,
        or as additional features for the Transformer to process along with the sequence.
        For simplicity, we'll create a sequence where each element is just a price,
        and then we'll concatenate the normalized cash and crypto *after* the Transformer
        processes the sequence, or incorporate them into the final linear layer.

        Let's adjust for the Transformer: The state will be a sequence of `window_size`
        data points. Each data point will be `[price, normalized_cash, normalized_crypto]`
        for that specific point in time. This is more complex because cash/crypto are
        portfolio-level, not price-level.

        Alternative and simpler for first Transformer attempt:
        State is `[price_t-k, ..., price_t, normalized_cash_t, normalized_crypto_t]`.
        The Transformer will operate on `[price_t-k, ..., price_t]` and we'll feed
        `normalized_cash_t, normalized_crypto_t` as additional inputs to the final
        decision layer of the Transformer.

        Let's stick to the previous state concatenation, but interpret the first `window_size`
        elements as the sequence for the Transformer, and the last 2 elements as auxiliary
        features fed *alongside* the Transformer output to a final dense layer.
        This means the `input_dim` for the Transformer will be `1` (price), and then
        the final layer will combine the Transformer's output with the 2 auxiliary features.
        This requires changing the `TransformerDQN` model.

        Simpler Transformer state: `price_window` only for the Transformer.
        Normalized cash/crypto will be appended to the final output of the transformer
        before the final linear layer (or handled as separate inputs to a fusion layer).
        For this simplified change, let's treat the state as a sequence of prices,
        and then normalize cash/crypto as separate features that get concatenated
        *after* the Transformer's encoding of the price sequence.

        So, `_get_state` returns `(price_window, normalized_cash, normalized_crypto)`.
        The `TransformerDQN` will take `price_window` and its `decoder` will also accept
        `normalized_cash` and `normalized_crypto`.
        """
        if self.current_step < self.window_size:
            price_window = np.zeros(self.window_size, dtype=np.float32)
        elif self.current_step >= len(self.prices):
            price_window = np.zeros(self.window_size, dtype=np.float32)
        else:
            price_window = np.array(self.prices[self.current_step - self.window_size:self.current_step], dtype=np.float32)

        # Normalize cash and crypto
        normalized_cash = self.cash / self.initial_cash
        # Avoid division by zero and handle very small prices
        max_possible_crypto = self.initial_cash / (self.min_price_overall + 1e-9) if self.min_price_overall > 0 else self.initial_cash / 1e-9
        normalized_crypto = self.crypto / (max_possible_crypto + 1e-9)

        # The state for the environment will still be a single concatenated array.
        # The Transformer model will then parse this into its sequence and auxiliary inputs.
        state = np.concatenate((price_window, [normalized_cash, normalized_crypto]))
        return state

    def step(self, action):
        """
        Takes an action in the environment and returns the next state, reward, and done status.
        Actions: 0 (Hold), 1-4 (Buy percentages), 5-8 (Sell percentages)
        """
        if self.current_step >= len(self.prices) - 1:
            return self._get_state(), 0, True

        price = self.prices[self.current_step]
        portfolio_value_before_action = self.cash + self.crypto * price

        # Define percentages for buy/sell actions
        percentages = [0.25, 0.50, 0.75, 1.00] # 25%, 50%, 75%, 100%

        if action >= 1 and action <= 4:  # Buy actions (1 to 4)
            buy_percent_idx = action - 1
            buy_percentage = percentages[buy_percent_idx]

            amount_to_spend = self.cash * buy_percentage

            if price * (1 + self.fee_percent) > 0:
                amount_to_buy_crypto = amount_to_spend / (price * (1 + self.fee_percent))
            else:
                amount_to_buy_crypto = 0

            if self.cash > 0 and amount_to_buy_crypto > 1e-9: # Small epsilon to avoid tiny trades
                self.crypto += amount_to_buy_crypto
                self.cash -= (amount_to_buy_crypto * price * (1 + self.fee_percent)) # Deduct spent amount and fees
                self.cash = max(0, self.cash) # Ensure cash doesn't go negative

        elif action >= 5 and action <= 8:  # Sell actions (5 to 8)
            sell_percent_idx = action - 5
            sell_percentage = percentages[sell_percent_idx]

            amount_to_sell_crypto = self.crypto * sell_percentage

            if self.crypto > 1e-9 and amount_to_sell_crypto > 1e-9: # Small epsilon
                revenue = amount_to_sell_crypto * price * (1 - self.fee_percent)
                self.cash += revenue
                self.crypto -= amount_to_sell_crypto # Deduct only the sold crypto amount
                self.crypto = max(0, self.crypto) # Ensure crypto doesn't go negative

        # Action 0: Hold (do nothing)

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1

        current_price_for_value = self.prices[self.current_step - 1]
        if done:
            current_price_for_value = self.prices[-1]

        portfolio_value_after_action = self.cash + self.crypto * current_price_for_value
        reward = portfolio_value_after_action - portfolio_value_before_action

        self.portfolio_value_history.append(portfolio_value_after_action)
        self.action_history.append(action)
        self.price_history.append(price)

        next_state = self._get_state() if not done else np.zeros(self.window_size + 2, dtype=np.float32)

        return next_state, reward, done

# Helper function to find the latest checkpoint (remains the same)
def find_latest_checkpoint(checkpoint_dir):
    """
    Scans the checkpoint directory and returns the path to the latest checkpoint
    and its corresponding episode number.
    Returns (None, 0) if no checkpoints are found.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return None, 0

    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("dqn_btc_trader_v3_episode_") and f.endswith(".pth")]

    latest_episode = 0
    latest_checkpoint_path = None

    for f in files:
        match = re.search(r'episode_(\d+)\.pth', f)
        if match:
            episode_num = int(match.group(1))
            if episode_num > latest_episode:
                latest_episode = episode_num
                latest_checkpoint_path = os.path.join(checkpoint_dir, f)

    if latest_checkpoint_path:
        print(f"✅ Found latest checkpoint: {latest_checkpoint_path} from episode {latest_episode}")
    else:
        print("🔍 No previous checkpoints found. Starting training from scratch.")

    return latest_checkpoint_path, latest_episode


# 4. Training Function
def train_agent(prices, episodes=1000, window_size=WINDOW_SIZE):
    """
    Trains a Transformer-based DQN agent to trade cryptocurrency.
    """
    feature_dim_per_step = 1 # Just the price
    aux_features_dim = 2 # Normalized cash and crypto

    env = TradingEnv(prices, window_size=window_size)

    nhead = 1
    num_encoder_layers = 1
    dim_feedforward = 64

    model = TransformerDQN(feature_dim=feature_dim_per_step, nhead=nhead,
                           num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
                           output_dim=9)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    memory = deque(maxlen=5000)



    all_episode_final_profits = []
    
    # Define the path for the episode metrics CSV
    metrics_csv_path = os.path.join(CHECKPOINT_DIR, "episode_metrics.csv")

    # Load existing metrics if resuming training
    if os.path.exists(metrics_csv_path):
        episode_metrics_df = pd.read_csv(metrics_csv_path)
        episode_metrics = episode_metrics_df.to_dict('records')
        print(f"📊 Loaded existing episode metrics from {metrics_csv_path}")
    else:
        episode_metrics = [] # Initialize an empty list if file doesn't exist

    latest_checkpoint_path, start_episode = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_checkpoint_path:
        checkpoint = torch.load(latest_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode']
        print(f"Resuming training from episode {start_episode + 1} with epsilon {epsilon:.4f}")
    else:
        print("Starting training from episode 1.")
        start_episode = 0

    print("\nStarting agent training...")
    print(f"🔍 Starting episode loop with WINDOW_SIZE={window_size}, total prices: {len(prices)}")
    
    for ep in range(start_episode, episodes):
        state_np = env.reset()
        print(f"▶️ Episode {ep+1}: Reset complete. First state shape: {state_np.shape}, current step: {env.current_step}")

        price_sequence = torch.tensor(state_np[:window_size], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        aux_features = torch.tensor(state_np[window_size:], dtype=torch.float32).unsqueeze(0)

        total_episode_reward = 0
        done = False
        step_count = 0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(9)
            else:
                with torch.no_grad():
                    q_values = model(price_sequence, aux_features)
                    action = torch.argmax(q_values).item()

            next_state_np, reward, done = env.step(action)

            next_price_sequence = torch.tensor(next_state_np[:window_size], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            next_aux_features = torch.tensor(next_state_np[window_size:], dtype=torch.float32).unsqueeze(0)

            memory.append((price_sequence, aux_features, action, reward, next_price_sequence, next_aux_features, done))

            total_episode_reward += reward

            price_sequence = next_price_sequence
            aux_features = next_aux_features
            step_count += 1

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)

                price_sequences_batch, aux_features_batch, actions_batch, rewards_batch, next_price_sequences_batch, next_aux_features_batch, dones_batch = zip(*batch)

                price_sequences_batch = torch.cat(price_sequences_batch)
                aux_features_batch = torch.cat(aux_features_batch)
                next_price_sequences_batch = torch.cat(next_price_sequences_batch)
                next_aux_features_batch = torch.cat(next_aux_features_batch)

                actions_batch = torch.tensor(actions_batch, dtype=torch.long)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
                dones_batch = torch.tensor(dones_batch, dtype=torch.bool)

                q_values = model(price_sequences_batch, aux_features_batch)

                with torch.no_grad():
                    next_q_values = model(next_price_sequences_batch, next_aux_features_batch).max(1)[0]
                    next_q_values[dones_batch] = 0.0

                targets = q_values.clone()

                for i in range(batch_size):
                    targets[i, actions_batch[i]] = rewards_batch[i] + (gamma * next_q_values[i])

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        final_profit_this_episode = env.portfolio_value_history[-1] - env.initial_cash
        all_episode_final_profits.append(final_profit_this_episode)

        action_counts = Counter(env.action_history)
        buys = sum(action_counts.get(i, 0) for i in range(1, 5))
        sells = sum(action_counts.get(i, 0) for i in range(5, 9))
        holds = action_counts.get(0, 0)

        print(f"Episode {ep+1}/{episodes}, Final Profit: ${final_profit_this_episode:.2f}, Epsilon: {epsilon:.4f}, Steps: {step_count}, Buys: {buys}, Sells: {sells}, Holds: {holds}")

        # --- IMPORTANT CHANGE STARTS HERE ---
        # Create a DataFrame for the current episode's metrics
        current_episode_df = pd.DataFrame([{
            'episode': ep + 1,
            'final_profit': int(final_profit_this_episode),
            'buys': buys,
            'sells': sells,
            'holds': holds
        }])

        # Append to CSV
        if not os.path.exists(metrics_csv_path):
            current_episode_df.to_csv(metrics_csv_path, index=False)
        else:
            current_episode_df.to_csv(metrics_csv_path, mode='a', header=False, index=False)
        # --- IMPORTANT CHANGE ENDS HERE ---

        if (ep + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"dqn_btc_trader_v3_episode_{ep + 1}.pth")
            torch.save({
                'episode': ep + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon
            }, checkpoint_path)
            print(f"--- Model weights saved to {checkpoint_path} ---")

    final_model_save_path = os.path.join(CHECKPOINT_DIR, "dqn_btc_trader_v3_final.pth")
    torch.save({
        'episode': episodes,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon
    }, final_model_save_path)
    print(f"\n✅ Training complete. Final model saved to {final_model_save_path}")


    plt.figure(figsize=(10, 6))
    plt.plot(range(start_episode + 1, episodes + 1), all_episode_final_profits)
    plt.title("Episode Final Net Profit Over Time (from resumed point)")
    plt.xlabel("Episode")
    plt.ylabel("Final Net Profit ($)")
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', label='Break-even')
    plt.legend()
    plt.show()

    print("\n--- Testing trained agent's performance ---")
    test_env = TradingEnv(prices, window_size=window_size)
    model.eval()
    test_state_np = test_env.reset()
    test_price_sequence = torch.tensor(test_state_np[:window_size], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    test_aux_features = torch.tensor(test_state_np[window_size:], dtype=torch.float32).unsqueeze(0)


    test_total_profit = 0
    test_done = False

    while not test_done:
        with torch.no_grad():
            test_q_values = model(test_price_sequence, test_aux_features)
            action = torch.argmax(test_q_values).item()

        next_state_np, reward, test_done = test_env.step(action)
        test_price_sequence = torch.tensor(next_state_np[:window_size], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        test_aux_features = torch.tensor(next_state_np[window_size:], dtype=torch.float32).unsqueeze(0)

        test_total_profit = test_env.portfolio_value_history[-1] - test_env.initial_cash

    test_action_counts = Counter(test_env.action_history)
    test_buys = sum(test_action_counts.get(i, 0) for i in range(1, 5))
    test_sells = sum(test_action_counts.get(i, 0) for i in range(5, 9))
    test_holds = test_action_counts.get(0, 0)

    print(f"Test Run Final Profit: ${test_total_profit:.2f}, Buys: {test_buys}, Sells: {test_sells}, Holds: {test_holds}")

    history_df = pd.DataFrame({
        'step': range(len(test_env.portfolio_value_history)),
        'portfolio_value': test_env.portfolio_value_history,
        'action': [None] * (test_env.window_size) + test_env.action_history if test_env.action_history else []
    })

    actual_prices_for_plot = []
    if len(test_env.prices) >= test_env.window_size:
        actual_prices_for_plot.append(test_env.prices[test_env.window_size - 1])

    for i in range(len(test_env.action_history)):
        if (test_env.window_size + i) < len(test_env.prices):
            actual_prices_for_plot.append(test_env.prices[test_env.window_size + i])
        else:
            actual_prices_for_plot.append(test_env.prices[-1])

    history_df['price_at_step'] = actual_prices_for_plot[:len(history_df['portfolio_value'])]


    if not history_df.empty:
        plt.figure(figsize=(14, 8))
        plt.plot(history_df['step'], history_df['portfolio_value'], label='Portfolio Value ($)', color='blue')

        ax2 = plt.gca().twinx()
        ax2.plot(history_df['step'], history_df['price_at_step'], label='BTC Price ($)', color='orange', alpha=0.7, linestyle='--')
        ax2.set_ylabel("BTC Price ($)")

        if 'action' in history_df.columns and not history_df['action'].isnull().all():
            buy_actions_to_plot = [1, 2, 3, 4]
            sell_actions_to_plot = [5, 6, 7, 8]

            buys_plot = history_df[history_df['action'].isin(buy_actions_to_plot)]
            sells_plot = history_df[history_df['action'].isin(sell_actions_to_plot)]

            if not buys_plot.empty:
                plt.scatter(buys_plot['step'], history_df.loc[buys_plot.index, 'price_at_step'], marker='^', color='green', s=100, label='Buy', alpha=1, zorder=5)
            if not sells_plot.empty:
                plt.scatter(sells_plot['step'], history_df.loc[sells_plot.index, 'price_at_step'], marker='v', color='red', s=100, label='Sell', alpha=1, zorder=5)

        plt.title("Trained Agent Performance (Test Run)")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()
    else:
        print("Test history is empty, cannot plot test performance.")


# --- Main execution ---
if __name__ == "__main__":
    try:
        btc_prices = get_price_data()
        print("🧾 BTC price data loaded:", len(btc_prices))
    except ValueError as e:
        print(e)
        print("Exiting. Please ensure your CSV has a 'Close' column or check the data source.")
        exit()

    if len(btc_prices) < WINDOW_SIZE + 2:
        print(f"Error: Not enough historical data ({len(btc_prices)} points) for window size {WINDOW_SIZE} with additional features.")
        print("Please ensure your CSV file or Yahoo Finance download provides sufficient data.")
        exit()

    train_agent(btc_prices, episodes=num_episodes, window_size=WINDOW_SIZE)