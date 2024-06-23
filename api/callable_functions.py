import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from typing import Any
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import insert

user = 'root'
password = 'root'
host = '127.0.0.1'
port = 3306
database = 'chatbot'

def get_connection():
    return create_engine(
        url="mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        )
    )
    
engine = get_connection()
conn = engine.connect()

def GetOrderDetails(order_id):
    user_product = pd.read_sql(f"SELECT * FROM user_products WHERE order_id = '{order_id}'", conn)
    
    order_id = user_product['order_id'].iloc[0]
    description = user_product['description'].iloc[0]
    purchased_at = str(user_product['purchased_at'].iloc[0])
    price = str(user_product['price'].iloc[0])
    
    return f'''Its a product  with decsription {description} you purchase at {purchased_at} costed ₹{price}'''

def CancelOrder(order_id):
    user_product = pd.read_sql(f"SELECT * FROM user_products WHERE order_id = '{order_id}'", conn)
    
    expected_delivary = user_product.expected_delivary.iloc[0]

    return f'The product was delivered to on {str(expected_delivary)}, As this is delivered to you before 10 days, I am afraid we cant cancel this order'

def GetOrderStatus(order_id):
    return 'Order status'

def ReturnOrder(order_id):
    
    return 'Return order initiated, Our delivery partner will pick up the prduct within 2 days!'

def GetRefundStatus(order_id):
    return 'refund status'

def ReplaceOrder(order_id):
    return 'ReplaceOrder'