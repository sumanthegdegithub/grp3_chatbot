import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from typing import Any

import langchain
from langchain.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.agents import Tool, tool
from langchain.chains import LLMChain
from fastapi import APIRouter, HTTPException, FastAPI, Request
import ast
import mysql.connector
from api.llms import model_for_order_id, model_for_function_calling, prompt_template_for_function_calling, prompt_template
from api.callable_functions import  GetOrderDetails, CancelOrder, GetOrderStatus, GetRefundStatus, ReturnOrder, ReplaceOrder
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import insert
import re

app = FastAPI()
api_router = APIRouter()

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
    
#engine = get_connection()
#conn = engine.connect()

def get_orders(text):
    return re.findall("ord[\\d]{5}", str(text))
  
prompt_template = f"""
You are a seasoned customer support executive at a quick commerece company. your company is specialized in product delivary within 10 minutes. You can greet the user and explain the user that you can help with selecting users past purchases and ask user to start explaining the purchase or product. Always obey system messages.

Below is the purchase history of user, based on it you need to return correct order ids

purchase_history:
products_go_here

Begin!
"""

def get_user_chats(USER_ID, prompt_template = prompt_template):

    #user_products = pd.read_sql(f"SELECT * FROM user_products WHERE user_id = {USER_ID}", conn)
    user_products = pd.read_csv('user_products.csv')
    USER_ID = int(USER_ID)
    user_products = user_products[user_products.user_id == USER_ID]
    products = ''

    for u in range(len(user_products)):
        order_id = user_products['order_id'].iloc[u]
        description = user_products['description'].iloc[u]
        purchased_at = str(user_products['purchased_at'].iloc[u])
        
        description = f'''order_id: {str(order_id)}, description: {str(description)}, purchased_at : {purchased_at}'''
        
        products += description
        products += '\n '
    
    
    if products != '':
      prompt_template = prompt_template.replace('products_go_here', products)
    else:
      prompt_template = prompt_template.replace('products_go_here', 'User has not purchased any product yet')
    
    conv = pd.DataFrame({})
    '''
    conv_end_count = pd.read_sql(f"SELECT COUNT(*) as conv_count FROM user_chats WHERE user_id = {USER_ID} AND type = 'ConvEnded'", conn)

    if conv_end_count.conv_count.iloc[0] == 0:
        conv_count = pd.read_sql(f"SELECT COUNT(*) as conv_count FROM user_chats WHERE user_id = {USER_ID}", conn)
        if conv_count.conv_count.iloc[0] > 0:
          conv = pd.read_sql(f"SELECT * FROM user_chats WHERE user_id = {USER_ID}", conn)
    else:
        #Get time of latest conv end
        latest_conv_end = pd.read_sql(f"SELECT created_at FROM user_chats WHERE user_id = {USER_ID} AND type = 'ConvEnded' ORDER BY created_at DESC LIMIT 1", conn)
        #Get number of convs after last session
        conv_count = pd.read_sql(f"SELECT COUNT(*) as conv_count FROM user_chats WHERE user_id = {USER_ID} AND created_at > '{str(latest_conv_end.created_at.iloc[0])}'", conn)
        if conv_count.conv_count.iloc[0] > 0:
          #Get convs after last session
          conv = pd.read_sql(f"SELECT * FROM user_chats WHERE user_id = {USER_ID} AND created_at > '{str(latest_conv_end.created_at.iloc[0])}'", conn)
    '''
    conversations = []
    order_id = ''
    if len(conv) > 0:
      for c in range(len(conv)):
          message_type = conv['type'].iloc[c]
          message = conv['message'].iloc[c]
          order_id = conv['order_id'].iloc[0]
          
          if message_type == 'HumanMessage':
            conversations.append(HumanMessage(content=message))
          elif message_type == 'SystemMessage':
            conversations.append(SystemMessage(content=message))
          elif message_type == 'AIMessage':
            conversations.append(AIMessage(content=message))
    else:
      conversations.append(SystemMessage(content=prompt_template))
      
    return conversations, order_id

@api_router.get("/backend_call", status_code=200)
async def run_llm(USER_ID, text, user_selected_product = '', user_confirmation = '', func_to_call = ''):
    USER_ID = int(USER_ID)
    if text == '':
        return {'user_selected_product': '', 'orders' : None, 'message' : None, 'resp_type' : None, 'function_to_call' : None }
    
    user_chats, order_id = get_user_chats(USER_ID, prompt_template)
    confirmation = user_confirmation
    
    user_products = pd.read_csv('user_products.csv')
    user_products = user_products[user_products.user_id == USER_ID]
    
    if (user_selected_product == '') & (order_id != ''):
        user_selected_product = order_id
     
    if user_selected_product == '':
        
        order_ids = get_orders(text)
        
        if len(order_ids) > 0:
            if len(order_ids) > 1:
                #count = pd.read_sql(f"SELECT count(*) as order_count FROM user_products WHERE user_id = {USER_ID} AND order_id IN {tuple(order_ids)}", conn)
                #count = count.order_count.iloc[0]
                count = len(user_products[user_products.order_id.isin(order_ids)])
                if count > 0:
                    return {'user_selected_product': '', 'orders' : order_ids, 'message' : 'Please confirm a product you want help with if its listed below, else try to describe', 'resp_type' : None, 'function_to_call' : None }
            elif len(order_ids) == 1:
                user_selected_product = order_ids[0]
                #user_products_count = pd.read_sql(f"SELECT COUNT(*) as product_count FROM user_products WHERE user_id = {USER_ID} AND order_id = '{user_selected_product}'", conn)
                #user_products_count = user_products_count.product_count.iloc[0]
                print(order_ids, type(order_ids), user_products.dtypes, user_products)
                user_products_count = len(user_products[user_products.order_id.astype(str).isin(order_ids)])
                if user_products_count == 0:
                    return {'user_selected_product': '', 'orders' : [], 'message' : f'The order Id {user_selected_product} is invalid, please provide a correct order id by checking your purchase history', 'resp_type' : None, 'function_to_call' : None }
                else:
                    return {'user_selected_product': '', 'orders' : order_ids, 'message' : 'Please confirm a product you want help with if its listed below, else try to describe', 'resp_type' : None, 'function_to_call' : None }
        else:
            user_chats.append(HumanMessage(content=text))
            ai_resp = model_for_order_id.invoke(user_chats)
            cont = ast.literal_eval(ai_resp.content)
            first_key = list(cont.keys())[0]
            
            
            msg = cont[first_key]
            order_ids = get_orders(msg)
            
            if len(order_ids) > 0:
                if len(order_ids) > 1:
                    #count = pd.read_sql(f"SELECT count(*) as order_count FROM user_products WHERE user_id = {USER_ID} AND order_id IN {tuple(order_ids)}", conn)
                    #count = count.order_count.iloc[0]
                    count = len(user_products[user_products.order_id.isin(order_ids)])
                    if count > 0:
                        return {'user_selected_product': '', 'orders' : order_ids, 'message' : 'Please confirm a product you want help with if its listed below, else try to describe', 'resp_type' : None, 'function_to_call' : None }
                elif len(order_ids) == 1:
                    user_selected_product = order_ids[0]
                    #user_products_count = pd.read_sql(f"SELECT COUNT(*) as product_count FROM user_products WHERE user_id = {USER_ID} AND order_id = '{user_selected_product}'", conn)
                    #user_products_count = user_products_count.product_count.iloc[0]
                    user_products_count = len(user_products[user_products.order_id.isin(order_ids)])
                    if user_products_count == 0:
                        return {'user_selected_product': '', 'orders' : [], 'message' : f'The order Id {user_selected_product} is invalid, please provide a correct order id by checking your purchase history', 'resp_type' : None, 'function_to_call' : None }
                    else:
                        return {'user_selected_product': '', 'orders' : order_ids, 'message' : 'Please confirm a product you want help with if its listed below, else try to describe', 'resp_type' : None, 'function_to_call' : None }
            else:
                return {'user_selected_product': '', 'orders' : [], 'message' :  msg, 'resp_type' : None, 'function_to_call' : None }
            
    
    #LLM 2 will me called from here
    else:
        if confirmation == '':
        
            conversations = []
            conversations.append(SystemMessage(content=prompt_template_for_function_calling))
            conversations.append(HumanMessage(content=f'my order ID is {user_selected_product} {text}'))
            
            ai_resp = model_for_function_calling.invoke(conversations)
            
            resp_type = None
            msg = ''
            if 'tool_calls' in str(ai_resp):
                tool_call = ai_resp.tool_calls[0]
                tool_name = tool_call.get('name', '')
                
                if tool_name in ['GetOrderDetails', 'CancelOrder', 'GetOrderStatus', 'ReturnOrder', 'GetRefundStatus', 'ReplaceOrder']:
                    if tool_name in ['CancelOrder', 'ReturnOrder', 'ReplaceOrder']:
                        resp_type = 'get_cofirmation'
                        return {'user_selected_product': user_selected_product, 'orders' : [], 'message' :  msg, 'resp_type' : resp_type, 'function_to_call' : tool_name }
                    else:
                        function_name = globals()[tool_name]
                        msg = function_name(user_selected_product)
                        resp_type = 'tool_msg'
                        return {'user_selected_product': user_selected_product, 'orders' : [], 'message' :  msg, 'resp_type' : resp_type, 'function_to_call' : None }
                elif  tool_name ==  '__conversational_response':
                    args = tool_call.get('args', {})
                    msg = args.get('response', '')
                    return {'user_selected_product': user_selected_product, 'orders' : [], 'message' :  msg, 'resp_type' : None, 'function_to_call' : None }
                else:
                    msg = 'Could you please rephrase?'
                    return {'user_selected_product': user_selected_product, 'orders' : [], 'message' :  msg, 'resp_type' : None, 'function_to_call' : None }
            else:
              return {'user_selected_product': user_selected_product, 'orders' : [], 'message' :  ai_resp.content, 'resp_type' : None, 'function_to_call' : None }
        else:
            if confirmation == 'true':
                function_name = globals()[func_to_call]
                msg = function_name(user_selected_product)
                resp_type = 'tool_msg'
                return {'user_selected_product': user_selected_product, 'orders' : [], 'message' :  msg, 'resp_type' : resp_type, 'function_to_call' : None }
            elif confirmation == 'false':
                msg = 'Please tell me how I can help with this order'
                return {'user_selected_product': user_selected_product, 'orders' : [], 'message' :  msg, 'resp_type' : resp_type, 'function_to_call' : None }

app.include_router(api_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
          
  
        
  
  
  
  
  