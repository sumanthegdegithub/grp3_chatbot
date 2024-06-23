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
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import Tool, tool
from langchain.chains import LLMChain




model_for_order_id = ChatOllama(
    model="phi3",
    keep_alive=-1,
    format="json",
    temperature = 0.0,
    cache = False,
    top_k = 5
    )

function_calling_model = OllamaFunctions(
    model="phi3",
    keep_alive=-1,
    format="json",
    temperature = 0.0,
    cache = False,
    top_k = 5
    )


model_for_function_calling = function_calling_model.bind_tools(
    tools=[
        {
            "name": "GetOrderDetails",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "order id in string like ord12345",
                    },
                }
            },
        },
         {
            "name": "CancelOrder",
            "description": "Call this function to cancel an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "order id in string like ord12345",
                    },
                }
            },
        },
        {
            "name": "GetOrderStatus",
            "description": "Get order status, This function is to know delivary status of order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "order id in string like ord12345",
                    },
                }
            },
        },
        {
            "name": "GetRefundStatus",
            "description": "Get refund status, This function is to know refund status of order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "order id in string like ord12345",
                    },
                }
            },
        },
        {
            "name": "ReturnOrder",
            "description": "Return an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "order id in string like ord12345",
                    },
                }
            },
        },
        {
            "name": "ReplaceOrder",
            "description": "Replace an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "order id in string like ord12345",
                    },
                }
            },
        },
        {
            "name": "__conversational_response",
            "description": (
                "Respond conversationally if no other tools should be called for a given query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Conversational response to the user.",
                    },
                },
                "required": ["response"],
            },
        },

    ],
    functions_to_call=["__conversational_response", "GetOrderDetails", "CancelOrder", "GetOrderStatus", "GetRefundStatus", "ReturnOrder", "ReplaceOrder"]
)


prompt_template_for_function_calling = """
You are a seasoned customer support executive at a e commerece company. You can greet the user and explain how you can help. Always obey system messages.
You should choose right tool based on users query
You have access to following tools, you should always choose atleast one of them.

__conversational_response(response: str) -> str - Respond conversationally if no other tools should be called for a given query or you dont have enough data to use any other functions
GetOrderDetails() -> str - Get order details.
CancelOrder() -> str - call this function to cancel an order.
GetOrderStatus() -> str - Get order status, This function is to know delivary status of order.
GetRefundStatus() -> str - Get refund status, This function is to know refund status of order.
ReturnOrder() -> str - Return an order.
ReplaceOrder() -> str - Replace an order.

Use the following format:

Question: the customer query you must support
Thought: Think what is the appropriate action to take.
Tool: the action to take, should be one of [__conversational_response, GetOrderDetails, CancelOrder, GetOrderStatus, GetRefundStatus, ReturnOrder, ReplaceOrder], you should always return atleast one of these tools
Tool Input: the input to the action, note for __conversational_response the input should be your response

Begin!
"""

prompt_template = f"""
You are a seasoned customer support executive at a quick commerece company. your company is specialized in product delivary within 10 minutes. You can greet the user and explain the user that you can help with selecting users past purchases and ask user to start explaining the purchase or product. Always obey system messages.

Below is the purchase history of user, based on it you need to return correct order ids

purchase_history:
products_go_here

Begin!
"""