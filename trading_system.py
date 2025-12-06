import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
import logging
from typing import List, Dict, Optional, Tuple, Any, Union

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingSignalSystem:
    def __init__(
        self, 
        account_size: float = 100000.0, 
        max_loss_pct: float = 2.0, 
        cooldown_hours: float = 4.0, 
        position_size_pct: float = 10.0,
        take_profit_pct: float = 0.015,
        time_exit_hours: float = 3.0,
        transaction_cost_pct: float = 0.001,  # 0.1% per trade
        model_params: Optional[Dict[str, Any]] = None
    ):  
        self.model = None
        self.feature_cols: List[str] = []
        self.account_size = account_size
        self.max_loss_pct = max_loss_pct
        self.cooldown_hours = cooldown_hours
        self.position_size_pct = position_size_pct
        self.take_profit_pct = take_profit_pct
        self.time_exit_hours = time_exit_hours
        self.transaction_cost_pct = transaction_cost_pct
        
        # model parameters
        self.model_params = model_params or {
            'n_estimators': 200,
            'max_depth': 5, 
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'gamma': 0.1, 
            'random_state': 42,
            'eval_metric': 'auc'
        }
        
        # Track actual position state
        self.current_position: Optional[Dict[str, Any]] = None  # {'entry_price', 'entry_time', 'shares', 'stop_loss'}
        self.last_trade_time: Optional[datetime] = None
        self.pnl_history: List[float] = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # === Returns ===
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Return lags
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        # === Rolling statistics on RETURNS ===
        for window in [5, 10, 20]:
            df[f'return_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'return_std_{window}'] = df['returns'].rolling(window).std()
        
        # === RSI ===
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._compute_rsi(df['close'], period)
        
        # === Volume ratios ===
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        volume_sma_safe = df['volume_sma_20'].replace(0, np.nan)
        df['volume_ratio'] = (df['volume'] / volume_sma_safe).clip(0, 10)  # Cap at 10x
        df['volume_change'] = df['volume'].pct_change().clip(-0.99, 10)
        
        # === Price vs MA ===
        for window in [10, 20, 50]:
            sma = df['close'].rolling(window).mean()
            sma_safe = sma.replace(0, np.nan)
            df[f'price_vs_sma_{window}'] = ((df['close'] - sma) / sma_safe).clip(-0.5, 0.5)
        
        # === Volatility ===
        df['atr_14'] = self._compute_atr(df, 14)
        close_safe = df['close'].replace(0, np.nan)
        df['atr_pct'] = (df['atr_14'] / close_safe).clip(0, 0.5)
        
        # === MACD ===
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd_raw = (exp1 - exp2) / close_safe
        df['macd'] = macd_raw.clip(-0.1, 0.1)
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = (df['macd'] - df['macd_signal']).clip(-0.05, 0.05)
        
        # === Momentum ===
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period).clip(-0.5, 0.5)
        
        # === High-Low range ===
        df['hl_range'] = ((df['high'] - df['low']) / close_safe).clip(0, 0.2)
        
        # Replace any remaining inf/nan
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        loss_safe = loss.replace(0, 1e-9)
        rs = gain / loss_safe
        rsi = 100 - (100 / (1 + rs))
        
        # Clip to valid range
        return rsi.clip(0, 100)
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def prepare_training_data(self, df: pd.DataFrame, forward_periods: int = 3, threshold: float = 0.008) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Create labels with safeguards"""
        df = self.create_features(df)
        
        # Target: future return
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Drop rows with NaN
        df = df.dropna()
        
        # Feature columns
        exclude_cols = ['target', 'future_return', 'open', 'high', 'low', 'close', 
                       'volume', 'timestamp', 'volume_sma_20', 'atr_14']
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Verify no price levels leaked in
        for col in self.feature_cols:
            if 'close_lag' in col or 'price_lag' in col or 'sma_' in col and 'vs' not in col:
                raise ValueError(f"Price level feature detected: {col}. Remove it!")
        
        X = df[self.feature_cols]
        y = df['target']
        
        return X, y, df
    
    def walk_forward_backtest(self, df: pd.DataFrame, n_splits: int = 5, forward_periods: int = 3, threshold: float = 0.008) -> pd.DataFrame:
        """Walk-forward validation with PnL tracking"""
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
                logger.warning(f"Fold {fold}: No positive samples in train, skipping")
                continue
            
            positive_ratio = y_train.sum() / len(y_train)
            scale_pos_weight = min((1 - positive_ratio) / positive_ratio, 50)  # Cap at 50
            
            # Train model
            params = self.model_params.copy()
            params['scale_pos_weight'] = scale_pos_weight
            
            model = xgb.XGBClassifier(**params)
            
            model.fit(X_train, y_train, verbose=False)
            
            # Predict on test
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
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
    
    def _simulate_trades(self, df_segment: pd.DataFrame, probas: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """Simulate actual trades with entry/exit and transaction costs"""
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
            
            # Exit logic
            elif position is not None:
                exit_row = df_segment.iloc[i]
                
                # Check stop loss (2 ATR)
                atr = self._compute_atr(df_segment.iloc[:i+1].tail(20), 14).iloc[-1]
                stop_loss = position['entry_price'] - (2 * atr)
                
                # Exit conditions
                is_stop_loss = exit_row['low'] <= stop_loss
                is_take_profit = (exit_row['high'] - position['entry_price']) / position['entry_price'] >= self.take_profit_pct
                is_time_exit = (i - position['entry_idx']) >= self.time_exit_hours
                
                if is_stop_loss or is_take_profit or is_time_exit:
                    exit_price = exit_row['close']
                    if is_stop_loss:
                        exit_price = min(exit_row['close'], stop_loss)
                    elif is_take_profit:
                        exit_price = position['entry_price'] * (1 + self.take_profit_pct)
                        
                    # Calculate PnL with transaction costs
                    # Cost is applied on both entry and exit
                    gross_pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    net_pnl_pct = gross_pnl_pct - (2 * self.transaction_cost_pct)
                    
                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': net_pnl_pct,
                        'bars_held': i - position['entry_idx'],
                        'reason': 'stop_loss' if is_stop_loss else ('take_profit' if is_take_profit else 'time_exit')
                    })
                    position = None
        
        return trades
    
    def _compute_sharpe(self, returns: List[float], periods_per_year: float = 252*6.5) -> float:
        """Compute Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        returns_arr = np.array(returns)
        return np.mean(returns_arr) / (np.std(returns_arr) + 1e-9) * np.sqrt(periods_per_year)
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        """Train final model on most recent data"""
        X, y, full_df = self.prepare_training_data(df)
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Positive signals in train: {y_train.sum()} ({y_train.mean():.1%})")
        
        # Guard against empty positive class
        if y_train.sum() == 0:
            raise ValueError("No positive samples in training data. Lower threshold or increase data.")
        
        positive_ratio = y_train.sum() / len(y_train)
        scale_pos_weight = min((1 - positive_ratio) / positive_ratio, 50)
        
        logger.info(f"Applying class balance: scale_pos_weight={scale_pos_weight:.1f}")
        
        params = self.model_params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        
        self.model = xgb.XGBClassifier(**params)
        
        self.model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        logger.info("\n=== Model Performance ===")
        logger.info(f"Probability range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
        logger.info(f"Mean probability: {y_pred_proba.mean():.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def generate_signal(self, current_data: pd.DataFrame, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate signal with REAL risk controls"""
        if self.model is None:
            return {'error': 'Model not trained'}
        
        timestamp = timestamp or datetime.now()
        
        # Check cooldown
        if self.last_trade_time:
            hours_since_last = (timestamp - self.last_trade_time).total_seconds() / 3600
            if hours_since_last < self.cooldown_hours:
                return {
                    'signal': 'HOLD',
                    'reason': f'Cooldown: {hours_since_last:.1f}h / {self.cooldown_hours}h since last trade',
                    'probability': 0.0,
                    'timestamp': timestamp
                }
        
        # Check if already in position
        if self.current_position:
            return {
                'signal': 'HOLD',
                'reason': f'Already in position since {self.current_position["entry_time"]}',
                'probability': 0.0,
                'timestamp': timestamp
            }
        
        # Ensure minimum data for features
        min_bars = 60
        if len(current_data) < min_bars:
            return {
                'signal': 'HOLD',
                'reason': f'Insufficient data: {len(current_data)} bars (need {min_bars})',
                'probability': 0.0,
                'timestamp': timestamp
            }
        
        # Check VIX / Fear & Greed
        current_vix = current_data['vix'].iloc[-1] if 'vix' in current_data.columns else None
        if current_vix and current_vix < 15:
             # VIX < 15 implies "Extreme Greed" / Complacency -> Higher risk of pullback
             # We might want to be stricter here
             pass
        
        # Create features
        df_feat = self.create_features(current_data)
        df_feat = df_feat.dropna()
        
        if len(df_feat) == 0:
            return {
                'signal': 'HOLD',
                'reason': 'Features contain NaN',
                'probability': 0.0,
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
            
        # Sentiment Filter (VIX)
        sentiment_boost = 0.0
        sentiment_reason = ""
        
        if current_vix:
            if current_vix > 30:
                sentiment_boost = 0.05
                sentiment_reason = f" (Boosted by Fear VIX={current_vix:.1f})"
            elif current_vix < 15:
                sentiment_boost = -0.10
                sentiment_reason = f" (Penalized by Greed VIX={current_vix:.1f})"
        
        final_proba = proba + sentiment_boost
        
        # Decision
        signal = 'BUY' if final_proba > 0.45 and shares > 0 else 'HOLD'
        
        if signal == 'BUY':
            reason = f"Model confidence {proba:.1%}{sentiment_reason}, RSI={df_feat['rsi_14'].iloc[-1]:.0f}"
        else:
            reason = f"Low confidence ({final_proba:.1%}){sentiment_reason} or insufficient capital"
        
        return {
            'signal': signal,
            'probability': proba,  # Return raw probability
            'adjusted_probability': final_proba,
            'reason': reason,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'shares': shares,
            'position_value': position_value,
            'risk_dollars': risk_dollars,
            'timestamp': timestamp,
            'vix': current_vix
        }
    
    def execute_signal(self, signal_result: Dict[str, Any]):
        """Execute trade and update position tracking"""
        if signal_result['signal'] == 'BUY':
            self.current_position = {
                'entry_price': signal_result['current_price'],
                'entry_time': signal_result['timestamp'],
                'shares': signal_result['shares'],
                'stop_loss': signal_result['stop_loss']
            }
            self.last_trade_time = signal_result['timestamp']
            logger.info(f"ENTERED: {signal_result['shares']} shares @ ${signal_result['current_price']:.2f}")
        
    def check_exit(self, current_data: pd.DataFrame, timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Check if we should exit current position"""
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
        elif pnl_pct >= self.take_profit_pct:
            should_exit = True
            reason = "Take profit target"
        elif (timestamp - self.current_position['entry_time']).total_seconds() / 3600 >= self.time_exit_hours:
            should_exit = True
            reason = f"Time exit ({self.time_exit_hours} hours)"
        
        if should_exit:
            # Apply transaction costs on exit 
            # Note: For PnL reporting we should include both entry and exit costs
            # Gross PnL
            gross_pnl_dollars = (current_price - entry_price) * shares
            
            # Transaction costs (approximate as % of value)
            entry_cost = entry_price * shares * self.transaction_cost_pct
            exit_cost = current_price * shares * self.transaction_cost_pct
            total_cost = entry_cost + exit_cost
            
            net_pnl_dollars = gross_pnl_dollars - total_cost
            net_pnl_pct = net_pnl_dollars / (entry_price * shares)
            
            self.pnl_history.append(net_pnl_dollars)
            
            logger.info(f"EXITED: {reason} | PnL: ${net_pnl_dollars:.2f} ({net_pnl_pct:.2%})")
            
            result = {
                'action': 'EXIT',
                'reason': reason,
                'pnl_dollars': net_pnl_dollars,
                'pnl_pct': net_pnl_pct,
                'exit_price': current_price
            }
            
            self.current_position = None
            return result
        
        return None


# ==================== DATA LOADERS ====================

class DataLoader:
    """Universal data loader"""
    
    @staticmethod
    def from_yfinance(symbol: str = 'SPY', period: str = '2y', interval: str = '1h') -> pd.DataFrame:
        """Load from Yahoo Finance"""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Install: pip install yfinance")
        
        # Fetch Target Symbol
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Fetch VIX (Fear & Greed Proxy)
        vix = yf.Ticker("^VIX")
        df_vix = vix.history(period=period, interval=interval)
        
        # Ensure timezones match (convert to UTC)
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC')
        if df_vix.index.tz is not None:
            df_vix.index = df_vix.index.tz_convert('UTC')
            
        # Merge VIX using nearest timestamp match
        # This handles slight offsets between SPY and VIX data
        vix_aligned = df_vix['Close'].reindex(df.index, method='nearest', tolerance=pd.Timedelta('30m'))
        df['vix'] = vix_aligned
        
        # Forward fill VIX if missing (e.g. holidays)
        df['vix'] = df['vix'].ffill().bfill()
        
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
        
        # Ensure columns exist
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vix']
        return df[cols]


# ==================== MAIN ====================

if __name__ == "__main__":
    logger.info("=== Production Trading System ===\n")
    
    # Load data
    try:
        df = DataLoader.from_yfinance(symbol='SPY', period='2y', interval='1h')
        logger.info(f"✓ Loaded {len(df)} bars")
        logger.info(f"✓ Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        
        # Initialize with custom config
        system = TradingSignalSystem(
            account_size=100000,
            max_loss_pct=2.0,
            cooldown_hours=4,
            position_size_pct=10,
            take_profit_pct=0.015,
            transaction_cost_pct=0.001  # 0.1% per trade
        )
        
        # Walk-forward backtest
        logger.info("--- Walk-Forward Backtest (5 folds) ---")
        results = system.walk_forward_backtest(df, n_splits=5)
        logger.info("\nBacktest Results:")
        summary = results.groupby('threshold').agg({
            'n_trades': 'sum',
            'win_rate': 'mean',
            'avg_pnl': 'mean',
            'total_pnl': 'sum',
            'sharpe': 'mean'
        }).round(3)
        print(summary)  # Keep print for table output
        
        # Train final model
        logger.info("\n--- Training Final Model ---")
        importance = system.train(df)
        print("\n=== Top 10 Features ===")
        print(importance.head(10).to_string(index=False))
        
        # Generate signal
        logger.info("\n--- Latest Signal ---")
        signal = system.generate_signal(df.tail(100))
        
        print(f"\nSignal: {signal['signal']}")
        print(f"Probability: {signal['probability']:.1%}")
        if 'vix' in signal and signal['vix']:
             print(f"VIX Index: {signal['vix']:.2f}")
        print(f"Reason: {signal['reason']}")
        if signal['signal'] == 'BUY':
            print(f"Entry: ${signal['current_price']:.2f}")
            print(f"Stop: ${signal['stop_loss']:.2f}")
            print(f"Shares: {signal['shares']}")
            print(f"Position: ${signal['position_value']:.2f}")
            print(f" Risk: ${signal['risk_dollars']:.2f}")
        
        logger.info("\nSystem validated and ready!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)