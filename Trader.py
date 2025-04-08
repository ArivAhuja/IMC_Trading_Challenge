import json
from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        result = {}
        target_product = "RAINFOREST_RESIN" # isolate one at a time
        
        # ignore any non-target products for now by setting their orders to empty
        for product in state.order_depths:
            if product != target_product:
                result[product] = []
        
        # if rainforest resin is missing from the order depths, nothing to do
        if target_product not in state.order_depths:
            return result, 1, state.traderData
        
        order_depth = state.order_depths[target_product]
        
        # determine the mid-price from the order_depth.
        mid_price = None
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
        elif order_depth.sell_orders:
            mid_price = min(order_depth.sell_orders.keys())
        elif order_depth.buy_orders:
            mid_price = max(order_depth.buy_orders.keys())
        
        if mid_price is None:
            result[target_product] = []
            return result, 1, state.traderData
        
        # delegate to the separated trading logic which also updates the trader state (price history)
        orders, new_trader_data = compute_rainforest_orders( # call the fiunction for the respective asset
            target_product, order_depth, mid_price, state.traderData
        )
        result[target_product] = orders
        
        # Return orders, a conversions value (kept as 1), and the updated traderData
        return result, 1, new_trader_data



def compute_rainforest_orders(target_product, order_depth, mid_price, trader_data, window_size=20, base_quantity=10):
    """
    Maintains a history of mid prices in trader_data, calculates the z-score using a rolling window,
    and returns a list of orders for rainforest resin.
    
    Parameters:
      - target_product (str): Product name, expected "RAINFOREST_RESIN"
      - order_depth (OrderDepth): OrderDepth object for rainforest resin
      - mid_price (float): Latest mid price calculated from the order book
      - trader_data (str): JSON string storing persistent data (e.g., price history)
      - window_size (int): Number of data points for the rolling window
      - base_quantity (int): Base quantity to trade; will be scaled up based on z-score magnitude
      
    Returns:
      - orders (list): List of Order objects (could be empty if no signal)
      - new_trader_data (str): Updated trader_data as a JSON string
    """
    # Load price history from trader_data
    try:
        history = json.loads(trader_data)
        if "rainforest_prices" not in history:
            history["rainforest_prices"] = []
    except Exception:
        history = {"rainforest_prices": []}
    
    # Append the latest mid price to the history
    history["rainforest_prices"].append(mid_price)

    orders = []
    # Only act if we have enough data points to calculate statistics
    if len(history["rainforest_prices"]) >= window_size:
        recent_prices = history["rainforest_prices"][-window_size:]
        rolling_mean = sum(recent_prices) / window_size
        # Calculate standard deviation (population std)
        rolling_std = (sum((p - rolling_mean) ** 2 for p in recent_prices) / window_size) ** 0.5
        
        # Safeguard against a zero standard deviation
        z_score = (mid_price - rolling_mean) / rolling_std if rolling_std != 0 else 0

        print("RAINFOREST_RESIN mid_price:", mid_price,
              "Rolling Mean:", rolling_mean,
              "Rolling Std:", rolling_std,
              "z_score:", z_score)

        # Scale the order quantity based on the extremeness of the z_score
        # For example, if |z_score| is 3, the scale_factor becomes 3. Otherwise minimum factor is 1.
        scale_factor = max(1, int(abs(z_score) * 2))
        order_quantity = base_quantity * scale_factor

        # If oversold (z_score < -2) then buy (use best ask)
        if z_score < -1.5 and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            orders.append(Order(target_product, best_ask, order_quantity))
            print("BUY", order_quantity, "of", target_product, "at price", best_ask)
        # If overbought (z_score > 2) then sell (use best bid)
        elif z_score > 1.5 and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order(target_product, best_bid, -order_quantity))
            print("SELL", order_quantity, "of", target_product, "at price", best_bid)
    
    # Return orders list and the updated trader data (history)
    return orders, json.dumps(history)


def compute_model_based_orders(target_product, order_depth, mid_price, trader_data, lag=4, base_quantity=10, threshold=1.0, max_position=50):
    """
    Uses pre-trained AR model parameters to predict the next price. The function
    then makes trading decisions only when the predicted price deviates significantly 
    (beyond a threshold) from the current mid price. It also adjusts orders to move the 
    current position (tracked in trader_data) towards a market neutral stance.
    
    Parameters:
      - target_product (str): Name of the product to trade.
      - order_depth (OrderDepth): Market depth for the product.
      - mid_price (float): The current mid price computed from the order book.
      - trader_data (str): Persistent JSON string storing data such as price history,
                           current position, and model parameters.
      - lag (int): Number of lagged price values used for predictions.
      - base_quantity (int): Base order quantity.
      - threshold (float): Minimum price difference between the predicted and current 
                           mid price to consider a trade.
      - max_position (int): Maximum allowed net exposure.
      
    Returns:
      - orders (list): List of Order objects.
      - new_trader_data (str): Updated trader_data JSON string.
    """
    try:
        history = json.loads(trader_data)
    except Exception:
        history = {}
    
    # Initialize price history and current position if not present
    if "model_history" not in history:
        history["model_history"] = []
    if "position" not in history:
        history["position"] = 0

    # Store pre-trained model parameters if not already present.
    if "model_params" not in history:
        history["model_params"] = {
            "intercept": 0.39903304835706876,
            "coefficients": [0.81988331, 0.1248814, 0.07095404, -0.01592438]
        }
        
    # Append the current mid_price to the history
    history["model_history"].append(mid_price)
    
    orders = []
    
    # Only predict if we have enough data points for the lag
    if len(history["model_history"]) >= lag:
        recent_prices = history["model_history"][-lag:]
        intercept = history["model_params"]["intercept"]
        coefficients = history["model_params"]["coefficients"]
        
        # Predict the next price using the autoregressive model
        predicted_price = intercept + sum(c * p for c, p in zip(coefficients, recent_prices))
        print(f"Predicted price for {target_product}: {predicted_price}, Current mid price: {mid_price}")
        
        # Calculate the deviation between the predicted price and current mid price
        deviation = predicted_price - mid_price
        
        # Only trade if the deviation exceeds the threshold
        if abs(deviation) > threshold:
            # Calculate desired trade quantity based on the magnitude of the deviation
            # For example, scale the base quantity by the deviation (can be capped)
            order_quantity = int(base_quantity * abs(deviation))
            
            # Prevent order_quantity from being too high
            order_quantity = min(order_quantity, max_position)
            
            current_position = history["position"]
            
            # If predicted price is higher than mid price, you expect an upward move:
            # Ideally, you want to be net long. If your current position is too short, buy.
            if deviation > threshold and order_depth.sell_orders:
                desired_position = max_position  # target to be long, up to max_position
                trade_size = desired_position - current_position
                if trade_size > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    orders.append(Order(target_product, best_ask, trade_size))
                    print(f"BUY {trade_size} of {target_product} at price {best_ask}")
                    history["position"] += trade_size
            # If predicted price is lower than mid price, expect a downward move:
            # Aim for net short. If your current position is too long, sell.
            elif deviation < -threshold and order_depth.buy_orders:
                desired_position = -max_position  # target to be short, up to -max_position
                trade_size = current_position - desired_position
                if trade_size > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    orders.append(Order(target_product, best_bid, -trade_size))
                    print(f"SELL {trade_size} of {target_product} at price {best_bid}")
                    history["position"] -= trade_size
        else:
            # If no significant deviation is detected, then do not adjust your position
            print("No significant signal. Holding position.")
    
    return orders, json.dumps(history)