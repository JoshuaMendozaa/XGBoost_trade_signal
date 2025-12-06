"""Level-invariant trading signal system with backtesting and signal generation."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class TradingSignalSystem:
    """Model-based trading signals with risk controls."""
    def __init__(self, account_size=100000, max_loss_pct=2.0, cooldown_hours=4, position_size_pct=10):
        self.model = None
        self.feature_cols = None
        self.signal_threshold = 0.5  # Will be tuned during training
        self.account_size = account_size
        self.max_loss_pct = max_loss_pct
        self.cooldown_hours = cooldown_hours
        self.position_size_pct = position_size_pct

        # Track actual position state
        self.current_position = None  # {'entry_price', 'entry_time', 'shares', 'stop_loss'}
        self.last_trade_time = None
        self.pnl_history = []

    def create_features(self, df):
        """Create level-invariant features from OHLCV."""
        df = df.copy()

        # === Returns (not levels) ===
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Return lags (not price lags!)
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)

        # === Rolling statistics on RETURNS ===
        for window in [5, 10, 20]:
            df[f'return_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'return_std_{window}'] = df['returns'].rolling(window).std()

        # === RSI (normalized 0-100) ===
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._compute_rsi(df['close'], period)

        # === Volume ratios (not absolute volume) ===
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        volume_sma_safe = df['volume_sma_20'].replace(0, np.nan)
        df['volume_ratio'] = (df['volume'] / volume_sma_safe).clip(0, 10)  # Cap at 10x
        df['volume_change'] = df['volume'].pct_change().clip(-0.99, 10)

        # === Price vs MA (percentage distance, not absolute) ===
        for window in [10, 20, 50, 200]:
            sma = df['close'].rolling(window).mean()
            sma_safe = sma.replace(0, np.nan)
            df[f'price_vs_sma_{window}'] = ((df['close'] - sma) / sma_safe).clip(-0.5, 0.5)

        # === Volatility (normalized by price) ===
        df['atr_14'] = self._compute_atr(df, 14)
        close_safe = df['close'].replace(0, np.nan)
        df['atr_pct'] = (df['atr_14'] / close_safe).clip(0, 0.5)

        # === MACD (normalized) ===
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd_raw = (exp1 - exp2) / close_safe
        df['macd'] = macd_raw.clip(-0.1, 0.1)
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = (df['macd'] - df['macd_signal']).clip(-0.05, 0.05)

        # === Momentum ===
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period).clip(-0.5, 0.5)

        # === High-Low range (normalized) ===
        df['hl_range'] = ((df['high'] - df['low']) / close_safe).clip(0, 0.2)

        # === NEW: Bollinger Band position (0-1 scale) ===
        bb_period = 20
        bb_sma = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        df['bb_position'] = ((df['close'] - bb_lower) / bb_range).clip(0, 1)

        # === NEW: Stochastic oscillator ===
        stoch_period = 14
        lowest_low = df['low'].rolling(stoch_period).min()
        highest_high = df['high'].rolling(stoch_period).max()
        stoch_range = (highest_high - lowest_low).replace(0, np.nan)
        df['stoch_k'] = ((df['close'] - lowest_low) / stoch_range * 100).clip(0, 100)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # === NEW: Rate of change in volatility ===
        df['vol_roc'] = df['return_std_10'].pct_change(5).clip(-2, 2)

        # === NEW: Cumulative return streaks ===
        df['up_streak'] = (df['returns'] > 0).astype(int)
        df['up_streak'] = df['up_streak'].groupby((df['up_streak'] != df['up_streak'].shift()).cumsum()).cumsum()
        df['down_streak'] = (df['returns'] < 0).astype(int)
        df['down_streak'] = df['down_streak'].groupby((df['down_streak'] != df['down_streak'].shift()).cumsum()).cumsum()

        # === NEW: Volume-price divergence ===
        price_up = df['returns'] > 0
        volume_up = df['volume_change'] > 0
        df['vol_price_agree'] = (price_up == volume_up).astype(int)

        # === NEW: Distance from recent high/low ===
        df['dist_from_high_20'] = ((df['close'] - df['high'].rolling(20).max()) / close_safe).clip(-0.5, 0)
        df['dist_from_low_20'] = ((df['close'] - df['low'].rolling(20).min()) / close_safe).clip(0, 0.5)

        # === VIX features (if available) ===
        if 'vix' in df.columns:
            df['vix_level'] = df['vix'].clip(10, 80)  # Clip extreme values
            df['vix_sma_20'] = df['vix'].rolling(20).mean()
            vix_sma_safe = df['vix_sma_20'].replace(0, np.nan)
            df['vix_vs_sma'] = ((df['vix'] - df['vix_sma_20']) / vix_sma_safe).clip(-1, 1)
            df['vix_change'] = df['vix'].pct_change().clip(-0.5, 0.5)
            df['vix_high'] = (df['vix'] > 25).astype(int)  # Fear indicator

        # === ADX (trend strength) for regime detection ===
        df['adx'] = self._compute_adx(df, 14)
        df['trending'] = (df['adx'] > 25).astype(int)  # 1 = trending, 0 = choppy

        # Replace any remaining inf/nan
        df = df.replace([np.inf, -np.inf], np.nan)

        return df

    def _compute_rsi(self, prices, period=14):
        """Compute RSI (0-100)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero
        loss_safe = loss.replace(0, 1e-9)
        rs = gain / loss_safe
        rsi = 100 - (100 / (1 + rs))

        # Clip to valid range
        return rsi.clip(0, 100)

    def _compute_atr(self, df, period=14):
        """Compute ATR."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    def _compute_adx(self, df, period=14):
        """Compute ADX (Average Directional Index) for trend strength."""
        high = df['high']
        low = df['low']
        close = df['close']

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # When +DM > -DM, -DM = 0 and vice versa
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        # Smoothed averages
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(period).mean()

        return adx.clip(0, 100)

    def prepare_training_data(self, df, forward_periods=5, threshold=0.005):
        """Create features and labels for training."""
        df = self.create_features(df)

        # Target: future return
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)

        # Max Adverse Excursion (MAE): worst drawdown during holding period
        # This is what we'll predict with the risk model
        df['mae'] = 0.0
        for i in range(len(df) - forward_periods):
            entry_price = df['close'].iloc[i]
            future_lows = df['low'].iloc[i+1:i+forward_periods+1]
            if len(future_lows) > 0:
                min_price = future_lows.min()
                df.iloc[i, df.columns.get_loc('mae')] = (min_price - entry_price) / entry_price

        # Risk target: is MAE worse than -1%? (high risk trade)
        df['high_risk'] = (df['mae'] < -0.01).astype(int)

        # Drop rows with NaN
        df = df.dropna()

        # Feature columns (exclude everything non-numeric or target-related)
        exclude_cols = ['target', 'future_return', 'mae', 'high_risk', 'open', 'high', 'low', 'close',
                       'volume', 'timestamp', 'volume_sma_20', 'atr_14', 'vix', 'vix_sma_20']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Verify no price levels leaked in
        for col in self.feature_cols:
            if 'close_lag' in col or 'price_lag' in col or 'sma_' in col and 'vs' not in col:
                raise ValueError(f"Price level feature detected: {col}. Remove it!")

        X = df[self.feature_cols]
        y = df['target']

        return X, y, df

    def walk_forward_backtest(self, df, n_splits=5, forward_periods=5, threshold=0.005):
        """Walk-forward backtest with trade simulation."""
        X, y, full_df = self.prepare_training_data(df, forward_periods, threshold)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        all_results = []
        fold = 0

        for train_idx, test_idx in tscv.split(X):
            fold += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Check for empty positive class
            if y_train.sum() == 0:
                print(f"Fold {fold}: No positive samples in train, skipping")
                continue

            positive_ratio = y_train.sum() / len(y_train)
            scale_pos_weight = min((1 - positive_ratio) / positive_ratio, 50)  # Cap at 50

            # Train model with time-ordered validation for early stopping
            val_split_idx = int(len(X_train) * 0.8)
            X_tr, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
            y_tr, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
            if len(X_val) == 0:
                X_tr, y_tr = X_train, y_train
                X_val, y_val = X_test, y_test

            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_lambda=1.0,
                reg_alpha=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1
            )
            model.fit(X_tr, y_tr)

            # Predict on test
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Model accuracy metrics for this fold
            fold_accuracy = accuracy_score(y_test, y_pred)
            fold_precision = precision_score(y_test, y_pred, zero_division=0)
            fold_recall = recall_score(y_test, y_pred, zero_division=0)
            fold_auc = roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else 0

            print(f"Fold {fold}: Accuracy={fold_accuracy:.1%}, Precision={fold_precision:.1%}, Recall={fold_recall:.1%}, AUC={fold_auc:.3f}")

            # Simulate trading with multiple thresholds
            for threshold_val in [0.3, 0.4, 0.5, 0.6]:
                trades = self._simulate_trades(
                    full_df.iloc[test_idx],
                    y_pred_proba,
                    threshold_val
                )

                all_results.append({
                    'fold': fold,
                    'threshold': threshold_val,
                    'n_trades': len(trades),
                    'win_rate': np.mean([t['pnl'] > 0 for t in trades]) if trades else 0,
                    'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
                    'total_pnl': sum([t['pnl'] for t in trades]),
                    'sharpe': self._compute_sharpe([t['pnl'] for t in trades]) if trades else 0
                })

        results_df = pd.DataFrame(all_results)
        return results_df

    def _simulate_trades(self, df_segment, probas, threshold):
        """Simulate entries and exits on test data."""
        trades = []
        position = None

        for i in range(len(df_segment) - 5):  # Leave room for exit
            row = df_segment.iloc[i]
            proba = probas[i]

            # Entry logic
            if position is None and proba > threshold:
                position = {
                    'entry_price': row['close'],
                    'entry_idx': i
                }

            # Exit logic (after 3 bars or stop loss hit)
            elif position is not None:
                exit_row = df_segment.iloc[i]

                # Check stop loss (2 ATR) with guard for NaNs/insufficient history
                atr_series = self._compute_atr(df_segment.iloc[:i+1].tail(20), 14)
                atr = atr_series.iloc[-1] if len(atr_series) else np.nan
                stop_loss = None
                if pd.notna(atr) and atr > 0:
                    stop_loss = position['entry_price'] - (2 * atr)

                stop_hit = stop_loss is not None and exit_row['low'] <= stop_loss
                time_exit = (i - position['entry_idx']) >= 3

                if stop_hit or time_exit:
                    exit_price = min(exit_row['close'], stop_loss) if stop_hit else exit_row['close']
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']

                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl_pct,
                        'bars_held': i - position['entry_idx']
                    })
                    position = None

        return trades

    def _compute_sharpe(self, returns, periods_per_year=252*6.5):
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0
        returns = np.array(returns)
        return np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(periods_per_year)

    def train(self, df, val_size=0.15, test_size=0.15, forward_periods=5, threshold=0.005):
        """Train alpha model + risk model with combined filtering."""
        X, y, full_df = self.prepare_training_data(df, forward_periods, threshold)

        # Risk target
        y_risk = full_df['high_risk'].loc[X.index]

        # === CLEAR 3-WAY TIME-BASED SPLIT ===
        n = len(X)
        train_end = int(n * (1 - val_size - test_size))
        val_end = int(n * (1 - test_size))

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        test_df = full_df.iloc[val_end:]

        y_risk_train = y_risk.iloc[:train_end]
        y_risk_val = y_risk.iloc[train_end:val_end]
        y_risk_test = y_risk.iloc[val_end:]

        # Guard against empty positive class
        if y_train.sum() == 0:
            raise ValueError("No positive samples in training data.")

        # ==================== ALPHA MODEL ====================
        positive_ratio = y_train.sum() / len(y_train)
        scale_pos_weight = min((1 - positive_ratio) / positive_ratio, 50)

        self.model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_lambda=1.0, reg_alpha=0.1, scale_pos_weight=scale_pos_weight,
            random_state=42, verbose=-1
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # ==================== RISK MODEL ====================
        risk_ratio = y_risk_train.sum() / len(y_risk_train)
        risk_scale = min((1 - risk_ratio) / risk_ratio, 50) if risk_ratio > 0 else 1

        self.risk_model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            reg_lambda=1.0, reg_alpha=0.1, scale_pos_weight=risk_scale,
            random_state=42, verbose=-1
        )
        self.risk_model.fit(X_train, y_risk_train, eval_set=[(X_val, y_risk_val)])

        # ==================== FIND OPTIMAL THRESHOLDS ====================
        y_val_alpha = self.model.predict_proba(X_val)[:, 1]
        y_val_risk = self.risk_model.predict_proba(X_val)[:, 1]

        best_thresh, best_precision = 0.5, 0
        for thresh in np.arange(0.3, 0.8, 0.05):
            y_val_pred = (y_val_alpha > thresh).astype(int)
            n_signals = y_val_pred.sum()
            if n_signals >= 10:
                prec = precision_score(y_val, y_val_pred, zero_division=0)
                if prec > best_precision:
                    best_precision, best_thresh = prec, thresh

        self.signal_threshold = best_thresh
        self.risk_threshold = 0.5  # Reject if risk > 50%

        # ==================== TEST: ALPHA ONLY ====================
        y_test_alpha = self.model.predict_proba(X_test)[:, 1]
        y_test_risk = self.risk_model.predict_proba(X_test)[:, 1]

        alpha_pred = (y_test_alpha > best_thresh).astype(int)
        alpha_wins = ((alpha_pred == 1) & (y_test == 1)).sum()
        alpha_losses = ((alpha_pred == 1) & (y_test == 0)).sum()
        alpha_win_rate = precision_score(y_test, alpha_pred, zero_division=0)

        # ==================== TEST: ALPHA + RISK FILTER ====================
        combined_pred = ((y_test_alpha > best_thresh) & (y_test_risk < self.risk_threshold)).astype(int)
        combined_wins = ((combined_pred == 1) & (y_test == 1)).sum()
        combined_losses = ((combined_pred == 1) & (y_test == 0)).sum()
        combined_win_rate = precision_score(y_test, combined_pred, zero_division=0) if combined_pred.sum() > 0 else 0

        # ==================== TEST: ALPHA + RISK + REGIME ====================
        trending = test_df['trending'].values
        regime_pred = ((y_test_alpha > best_thresh) & (y_test_risk < self.risk_threshold) & (trending == 1)).astype(int)
        regime_wins = ((regime_pred == 1) & (y_test == 1)).sum()
        regime_losses = ((regime_pred == 1) & (y_test == 0)).sum()
        regime_win_rate = precision_score(y_test, regime_pred, zero_division=0) if regime_pred.sum() > 0 else 0

        # ==================== PRINT RESULTS ====================
        print(f"Prediction: SPY will rise {threshold:.1%}+ in next {forward_periods} bars")
        print(f"Test period: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")

        print(f"\n{'='*50}")
        print("ALPHA MODEL ONLY")
        print(f"{'='*50}")
        print(f"BUY signals: {alpha_pred.sum()} | Wins: {alpha_wins} | Losses: {alpha_losses}")
        print(f"Win rate: {alpha_win_rate:.1%}")
        if alpha_win_rate > 0 and alpha_win_rate < 1:
            print(f"Min R/R: {(1-alpha_win_rate)/alpha_win_rate:.1f}:1")

        print(f"\n{'='*50}")
        print("ALPHA + RISK FILTER (reject high drawdown)")
        print(f"{'='*50}")
        print(f"BUY signals: {combined_pred.sum()} | Wins: {combined_wins} | Losses: {combined_losses}")
        print(f"Win rate: {combined_win_rate:.1%}")
        if combined_win_rate > 0 and combined_win_rate < 1:
            print(f"Min R/R: {(1-combined_win_rate)/combined_win_rate:.1f}:1")

        print(f"\n{'='*50}")
        print("ALPHA + RISK + REGIME (only trending markets)")
        print(f"{'='*50}")
        print(f"BUY signals: {regime_pred.sum()} | Wins: {regime_wins} | Losses: {regime_losses}")
        print(f"Win rate: {regime_win_rate:.1%}")
        if regime_win_rate > 0 and regime_win_rate < 1:
            print(f"Min R/R: {(1-regime_win_rate)/regime_win_rate:.1f}:1")

        # ==================== VOLATILITY SIZING INFO ====================
        avg_atr = test_df['atr_pct'].mean()
        print(f"\n{'='*50}")
        print("POSITION SIZING (volatility-adjusted)")
        print(f"{'='*50}")
        print(f"Avg ATR: {avg_atr:.2%} of price")
        print(f"Suggested: Risk 1% of account per trade")
        print(f"Position size = (Account × 1%) / (Entry × ATR × 2)")
        print(f"Example: $100k account, $500 entry, 1% ATR")
        print(f"  Size = ($100k × 1%) / ($500 × 1% × 2) = 100 shares")

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    def generate_signal(self, current_data, timestamp=None):
        """Generate latest signal with sizing and risk."""
        if self.model is None:
            return {'error': 'Model not trained'}

        timestamp = timestamp or datetime.now()

        # Check cooldown (now actually works!)
        if self.last_trade_time:
            hours_since_last = (timestamp - self.last_trade_time).total_seconds() / 3600
            if hours_since_last < self.cooldown_hours:
                return {
                    'signal': 'HOLD',
                    'reason': f'Cooldown: {hours_since_last:.1f}h / {self.cooldown_hours}h since last trade',
                    'probability': 0,
                    'timestamp': timestamp
                }

        # Check if already in position
        if self.current_position:
            return {
                'signal': 'HOLD',
                'reason': f'Already in position since {self.current_position["entry_time"]}',
                'probability': 0,
                'timestamp': timestamp
            }

        # Ensure minimum data for features
        min_bars = 60
        if len(current_data) < min_bars:
            return {
                'signal': 'HOLD',
                'reason': f'Insufficient data: {len(current_data)} bars (need {min_bars})',
                'probability': 0,
                'timestamp': timestamp
            }

        # Create features
        df_feat = self.create_features(current_data)
        df_feat = df_feat.dropna()

        if len(df_feat) == 0:
            return {
                'signal': 'HOLD',
                'reason': 'Features contain NaN',
                'probability': 0,
                'timestamp': timestamp
            }

        latest = df_feat.iloc[-1:][self.feature_cols]

        # Predict
        proba = self.model.predict_proba(latest)[0, 1]

        # Calculate position size and risk
        current_price = current_data['close'].iloc[-1]
        atr = df_feat['atr_14'].iloc[-1]

        if pd.isna(atr) or atr <= 0:
            return {
                'signal': 'HOLD',
                'reason': 'Invalid ATR for risk calculation',
                'probability': proba,
                'timestamp': timestamp
            }

        stop_loss = current_price - (2 * atr)
        risk_per_share = current_price - stop_loss
        risk_dollars = (self.account_size * self.max_loss_pct / 100)
        shares = int(risk_dollars / risk_per_share)
        position_value = shares * current_price

        # Check if position size is reasonable
        max_position = self.account_size * (self.position_size_pct / 100)
        if position_value > max_position:
            shares = int(max_position / current_price)
            position_value = shares * current_price

        # Decision
        signal = 'BUY' if proba > self.signal_threshold and shares > 0 else 'HOLD'

        if signal == 'BUY':
            reason = f"Model confidence {proba:.1%}, RSI={df_feat['rsi_14'].iloc[-1]:.0f}"
        else:
            reason = f"Low confidence ({proba:.1%}) or insufficient capital"

        return {
            'signal': signal,
            'probability': proba,
            'reason': reason,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'shares': shares,
            'position_value': position_value,
            'risk_dollars': risk_dollars,
            'timestamp': timestamp
        }

    def execute_signal(self, signal_result):
        """Execute a BUY signal and update open-position tracking."""
        if signal_result['signal'] == 'BUY':
            self.current_position = {
                'entry_price': signal_result['current_price'],
                'entry_time': signal_result['timestamp'],
                'shares': signal_result['shares'],
                'stop_loss': signal_result['stop_loss']
            }
            self.last_trade_time = signal_result['timestamp']
            print(f"ENTERED: {signal_result['shares']} shares @ ${signal_result['current_price']:.2f}")

    def check_exit(self, current_data, timestamp=None):
        """Evaluate exit conditions for the current position if one exists."""
        if not self.current_position:
            return None

        timestamp = timestamp or datetime.now()
        current_price = current_data['close'].iloc[-1]
        entry_price = self.current_position['entry_price']
        stop_loss = self.current_position['stop_loss']
        shares = self.current_position['shares']

        # Check stop loss or take profit
        pnl_pct = (current_price - entry_price) / entry_price

        should_exit = False
        reason = ""

        if current_price <= stop_loss:
            should_exit = True
            reason = "Stop loss hit"
        elif pnl_pct >= 0.015:  # Take profit at 1.5%
            should_exit = True
            reason = "Take profit target"
        elif (timestamp - self.current_position['entry_time']).total_seconds() / 3600 >= 3:
            should_exit = True
            reason = "Time exit (3 hours)"

        if should_exit:
            pnl_dollars = (current_price - entry_price) * shares
            self.pnl_history.append(pnl_dollars)

            print(f"EXITED: {reason} | PnL: ${pnl_dollars:.2f} ({pnl_pct:.2%})")

            result = {
                'action': 'EXIT',
                'reason': reason,
                'pnl_dollars': pnl_dollars,
                'pnl_pct': pnl_pct,
                'exit_price': current_price
            }

            self.current_position = None
            return result

        return None


# ==================== DATA LOADERS ====================

class DataLoader:
    """Load market data."""

    @staticmethod
    def from_yfinance(symbol, period, interval, include_vix=True):
        """Load OHLCV from yfinance with optional VIX."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Install: pip install yfinance")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Add VIX data
        if include_vix:
            vix = yf.Ticker('^VIX')
            vix_df = vix.history(period=period, interval=interval)
            vix_df = vix_df.reset_index()
            vix_df = vix_df.rename(columns={
                'Date': 'timestamp',
                'Datetime': 'timestamp',
                'Close': 'vix'
            })
            vix_df = vix_df[['timestamp', 'vix']]

            # Merge on date (VIX timestamps may differ slightly)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            vix_df['date'] = pd.to_datetime(vix_df['timestamp']).dt.date
            df = df.merge(vix_df[['date', 'vix']], on='date', how='left')
            df = df.drop(columns=['date'])
            df['vix'] = df['vix'].ffill()  # Forward fill any gaps

        return df


# ==================== MAIN ====================

if __name__ == "__main__":
    # Load data
    df = DataLoader.from_yfinance(symbol='SPY', period='max', interval='1d')

    # Train model: 0.3% in 3 days (more trades)
    system = TradingSignalSystem()
    system.train(df, forward_periods=3, threshold=0.005)