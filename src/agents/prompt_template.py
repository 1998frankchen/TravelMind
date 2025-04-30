from typing import Dict, List, Optional  
from pydantic import BaseModel  

class ToolParameter(BaseModel):  
    """Tool parameter specification"""  
    name: str  
    type: str  
    description: str  
    required: bool = True  

class ToolTemplate:  
    def __init__(self,   
                 name: str,  
                 description: str,  
                 parameters: List[ToolParameter],  
                 call_template: str):  
        """  
        Args:  
            name: Tool name (English identifier)    
            description: Natural language description  
            parameters: Parameter list  
            call_template: Call template, e.g., google_search(query)    
        """  
        self.name = name  
        self.description = description  
        self.parameters = parameters  
        self.call_template = call_template  

class MyPromptTemplate:  
    def __init__(self):  
        self.tools: Dict[str, ToolTemplate] = {
            
        }  
        self.base_prompt = """You are a travel route planning assistant. Your task is to use appropriate tools to obtain travel-related information and plan the most accurate, comfortable, and efficient travel routes. Please follow these principles:
                        1. Use tools in logical order (e.g., check weather first, then book hotels)
                        2. Ensure parameter formats are correct (use YYYY-MM-DD format for dates)
                        3. Prioritize using the latest data
                        4. When user confirmation is needed, ask in natural language

                        You can use the following tools:"""  
        # Pre-register common tools  
        self.register_tool(self._build_hotel_template())  
        self.register_tool(self._build_plane_template())  
        self.register_tool(self._build_transportation_template())  
        self.register_tool(self._build_weather_template())
    
    def register_tool(self, tool: ToolTemplate):  
        """Register new tool"""    
        self.tools[tool.name] = tool  
        
    def get_tools(self)->str:
        return "\n".join(
            [f"{tool.name}: {tool.description}\nParameter: {[p.name for p in tool.parameters]}"
             for tool in self.tools.values()]
        )
        
    def get_tool_format(self):
        return """
        <thinking>Analyze question/problem and select tool</thinking>    
        <tool_call>{'{ToolName}'}(Parameter1=Value1, Parameter2=Value2)</tool_call>  
        """
    
    def generate_prompt(self, query: str, history: Optional[List] = None) -> str:  
        """Generate complete prompt"""  
        tools_desc = "\n".join(  
            [f"{tool.name}: {tool.description}\nParameter: {[p.name for p in tool.parameters]}"  
             for tool in self.tools.values()]  
        )  
        
        return f"""{self.base_prompt}  
        
                Available tool list:  
                {tools_desc}  

                
                Current conversation history:    
                {(history[:500] if len(history) > 500 else history) if history else "None"}    

                User question: {query}  
                
                You can only select one tool from the tool list to respond. Please strictly follow the tool call format below.  
                
                Please respond in the following format:    
                <thinking>Analyze question/problem and select tool</thinking>    
                <tool_call>{'{ToolName}'}(Parameter1=Value1, Parameter2=Value2)</tool_call>  
                
            
                """
    def get_tool_format(self):
        
        return f"""
                <thinking>Analyze question/problem and select tool</thinking>    
                <tool_call>{'{ToolName}'}(Parameter1=Value1, Parameter2=Value2)</tool_call>  
                """


    def get_tool_desc(self):
        tools_desc = "\n".join(  
            [f"{tool.name}: {tool.description}\nParameter: {[p.name for p in tool.parameters]}"  
             for tool in self.tools.values()]  
        )  
        return tools_desc
                
    def _build_hotel_template(self):  
        return ToolTemplate(  
            name="book_hotel",  
            description="Hotel booking service, supports filtering by location, budget and date",    
            parameters=[  
                ToolParameter(name="location", type="str", description="City name or specific address"),    
                ToolParameter(name="check_in", type="str", description="Check-in date (YYYY-MM-DD)"),  
                ToolParameter(name="check_out", type="str", description="Check-out date (YYYY-MM-DD)"),  
                ToolParameter(name="budget", type="int", description="Budget per night (RMB)", required=False),    
                ToolParameter(name="room_type", type="str", description="Room type requirement, e.g., king bed/twin bed", required=False)    
            ],  
            call_template="book_hotel(location={location}, check_in={check_in}, check_out={check_out})"  
        )  

    def _build_plane_template(self):  
        return ToolTemplate(  
            name="book_flight",  
            description="Flight ticket booking service, supports multi-city query and price comparison",    
            parameters=[  
                ToolParameter(name="departure", type="str", description="Departure city airport code (e.g., PEK)"),    
                ToolParameter(name="destination", type="str", description="Destination city airport code (e.g., SHA)"),    
                ToolParameter(name="date", type="str", description="Departure date (YYYY-MM-DD)"),    
                ToolParameter(name="seat_class", type="str", description="Seat class: economy/business/first", required=False)    
            ],  
            call_template="book_flight(departure={departure}, destination={destination}, date={date})"  
        )  

    def _build_transportation_template(self):  
        return ToolTemplate(  
            name="find_fastest_route",  
            description="Real-time transportation route planning, supports multiple transportation modes",    
            parameters=[  
                ToolParameter(name="start", type="str", description="Start coordinates or address"),    
                ToolParameter(name="end", type="str", description="End coordinates or address"),    
                ToolParameter(name="transport_type", type="str", description="Transportation mode: driving/walking/transit", required=False)    
            ],  
            call_template="find_fastest_route(start={start}, end={end})"  
        )  

    def _build_weather_template(self):  
        return ToolTemplate(  
            name="get_weather",  
            description="Multi-day weather forecast query service",    
            parameters=[  
                ToolParameter(name="location", type="str", description="City name or postal code"),    
                ToolParameter(name="date", type="str", description="Query date (YYYY-MM-DD)")  
            ],  
            call_template="get_weather(location={location}, date={date})"  
        )  

