import requests
import json
import re
from typing import Dict, Any, List
import os

from bs4 import BeautifulSoup
# from baidusearch.baidusearch import search
from urllib.parse import quote_plus
import time
import random

from datetime import datetime


from src.agents.prompt_template import MyPromptTemplate



# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'



class ToolDispatcher:
    def __init__(self):
        self.executors = {
            # "google_search": GoogleSearchExecutor(
            #     api_key=CUSTOM_SEARCH_API_KEY,
            #     search_engine_id=CUSTOM_SEARCH_ENGINE_ID
            # ),

            "get_weather": WeatherExecutor,
            "book_hotel": HotelExecutor,
            "book_flight": PlaneTicketExecutor,
            "find_fastest_route": TransportationExecutor,

        }

        self.prompt_template = MyPromptTemplate()

    def parse_tool_call(self, tool_str: str) -> Dict:
        """Parse tool call string"""
        pattern = r"(\w+)\((.*)\)"
        match = re.match(pattern, tool_str)
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse parameter key-value pairs
        args = {}
        for pair in re.findall(r"(\w+)=([^,]+)", args_str):
            key = pair[0]
            value = pair[1].strip("'")
            if re.match(r'^-?\d+$', value):  # Support negative integers
                value = int(value)
            args[key] = value

        return {"tool": tool_name, "args": args}

    def execute(self, tool_call: str) -> Dict:
        """Execute tool call"""
        parsed = self.parse_tool_call(tool_call)
        if not parsed:
            return {"error": "Invalid tool format"}

        executor = self.executors.get(parsed["tool"])
        if not executor:
            return {"error": "Tool not registered"}

        # Get tool parameter specification
        tool_template = self.prompt_template.tools.get(parsed["tool"])
        if not tool_template:
            return {"error": "Tool template not found"}


        # Parameter type verification
        for param in tool_template.parameters:
            if param.required and param.name not in parsed["args"]:
                return {"error": f"Missing required parameter: {param.name}"}
            if param.name in parsed["args"]:
                expected_type = param.type
                actual_value = parsed["args"][param.name]
                if not isinstance(actual_value, eval(expected_type)):
                    return {"error": f"Type mismatch for {param.name}, expected {expected_type}"}

        print( "parse_args = ", parsed["args"])

        # parsed["args"] = {"query":..., "max_results":...}
        return executor.execute(**parsed["args"])




class GoogleSearchExecutor:
    def __init__(self, api_key: str, search_engine_id: str):
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.api_key = api_key
        self.search_engine_id = search_engine_id

    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": max_results
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._parse_results(response.json())
        except Exception as e:
            return {"error": str(e)}

    def _parse_results(self, data: Dict) -> Dict:
        """ParseGoogle APIResponse"""
        return {
            "items": [{
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            } for item in data.get("items", [])]
        }


class WeatherExecutor:
    def __init__(self, api_key: str = ""):
        self.base_url = "https://api.weatherapi.com/v1/history.json"
        self.api_key = api_key  # API key required for actual use

    def execute(self, location: str, date: str) -> Dict[str, Any]:
        """Simulate weather data query"""
        try:
            # Validate date format
            datetime.strptime(date, "%Y-%m-%d")

            # Simulated data (should call actual API in production)
            return {
                "location": location,
                "date": date,
                "temperature": random.randint(15, 30),
                "condition": random.choice(["Sunny", "Cloudy", "Light Rain"]),
                "humidity": f"{random.randint(40, 80)}%"
            }
        except ValueError:
            return {"error": "Invalid date format, use YYYY-MM-DD"}
        except Exception as e:
            return {"error": str(e)}

class HotelExecutor:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key  # Hotel API key

    def execute(self, location: str, check_in: str, check_out: str,
               budget: int = None, room_type: str = None) -> Dict[str, Any]:
        """Hotel booking simulation"""
        try:
            # Parameter validation
            datetime.strptime(check_in, "%Y-%m-%d")
            datetime.strptime(check_out, "%Y-%m-%d")

            # Simulated response (should call actual API in production)
            hotels = [{
                "name": f"{location} Hotel Example {1}",
                "price": random.randint(300, 800),
                "room_type": room_type or "King Room",
                "rating": round(random.uniform(3.5, 5.0), 1)
            } for _ in range(3)]

            if budget:
                hotels = [h for h in hotels if h["price"] <= budget]

            return {"available_hotels": hotels}
        except ValueError:
            return {"error": "Invalid date format"}
        except Exception as e:
            return {"error": str(e)}

class PlaneTicketExecutor:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key  # Flight API key

    def execute(self, departure: str, destination: str, date: str,
               seat_class: str = "economy") -> Dict[str, Any]:
        """Flight search simulation"""
        try:
            datetime.strptime(date, "%Y-%m-%d")

            # Simulated flight data
            flights = [{
                "flight_no": f"{random.choice(['MU', 'CA', 'CZ'])}{random.randint(1000,9999)}",
                "departure_time": f"{random.randint(6,22)}:00",
                "price": random.randint(500, 2000),
                "seat_class": seat_class
            } for _ in range(3)]

            return {
                "departure": departure,
                "destination": destination,
                "flights": sorted(flights, key=lambda x: x["price"])
            }
        except ValueError:
            return {"error": "Invalid date format"}
        except Exception as e:
            return {"error": str(e)}

class TransportationExecutor:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key  # Map API key

    def execute(self, start: str, end: str,
               transport_type: str = "driving") -> Dict[str, Any]:
        """Transportation route planning simulation"""
        try:
            # Simulate different transportation methods
            base_time = random.randint(20, 60)
            times = {
                "driving": base_time,
                "walking": base_time * 4,
                "transit": base_time * 1.5
            }

            return {
                "start": start,
                "end": end,
                "duration": f"{times.get(transport_type, base_time)} minutes",
                "distance": f"{random.randint(5, 20)} km",
                "route": [
                    f"Depart from {start}",
                    "Follow simulated route",
                    f"Arrive at {end}"
                ]
            }
        except Exception as e:
            return {"error": str(e)}