
try:
   from src.models.model import TravelMind  
   # from src.ui.app import launch_ui  

   from src.finetune.sft_trainer import SFTTrainer

   from src.utils.utils import SFTArguments

   from src.configs.config import MODEL_PATH, DATA_PATH, SFT_MODEL_PATH

   from src.data.data_processor import TravelQAProcessor


   from src.agents.rag import RAG

   from src.agents.rag import CityRAG



   from src.agents.agent import MyAgent
   
   from src.agents.rag_dispatcher import RAGDispatcher
   
   
except Exception as e:
   print("Import error occurred, likely a version issue, but ignoring for now: ", str(e))
   print("==================================")

import argparse
import asyncio

PLAN_EXAMPLE = """

### **I. Pre-Departure Preparation**
1. **Visa Application**:
   - Ensure you have a valid US visa (B1/B2 tourist visa).
   - Schedule visa interview in advance, prepare required documents (passport, photos, bank statements, itinerary, etc.).

2. **Flight Booking**:
   - Book flights 1-2 months in advance, choose direct or connecting flights.
   - Recommended airlines: Air China (direct), Delta Airlines, United Airlines, etc.
   - Direct flight duration is about 13 hours, connecting flights may take 20+ hours.

3. **Accommodation Booking**:
   - Choose hotels or Airbnb based on budget, recommend staying in Midtown Manhattan (convenient transportation, close to major attractions).
   - Book in advance through platforms like Booking.com or Airbnb.

4. **Travel Insurance**:
   - Purchase insurance covering medical, baggage loss, flight delays, etc.

5. **Luggage Preparation**:
   - Prepare clothing based on season (NYC is cold in winter, hot in summer).
   - Bring power adapters (US voltage is 110V, plug types A/B).

---

### **II. Transportation Arrangements**
#### **1. Beijing to New York Flights**
   - **Direct Flights**: Depart from Beijing Capital International Airport (PEK) or Daxing International Airport (PKX), arrive at JFK or Newark Liberty International Airport (EWR).
   - **Connecting Flights**: Options include transfers in Tokyo, Seoul, or European cities (Paris, London).

#### **2. NYC Local Transportation**
   - **Airport to City**:
     - JFK Airport: Take AirTrain connecting to subway (E or A line), or taxi/rideshare (~$60-$80).
     - EWR Airport: Take AirTrain connecting to NJ Transit train, or taxi/rideshare (~$70-$90).
   - **City Transportation**:
     - Subway is the most convenient, recommend 7-day unlimited MetroCard (~$34).
     - Uber/Lyft rideshare services available.

---

### **III. New York Itinerary**
#### **Day 1: Arrival and Orientation**
   - Check into hotel after arrival, adjust to time difference.
   - Evening visit to **Times Square** to experience NYC nightlife.

#### **Day 2: Manhattan Classic Sights**
   - Morning: Visit **Statue of Liberty** (take ferry to Liberty Island).
   - Afternoon: Explore **Wall Street**, **One World Observatory**.
   - Evening: Walk across **Brooklyn Bridge**, enjoy Manhattan night views.

#### **Day 3: Culture and Arts Tour**
   - Morning: Visit **The Metropolitan Museum of Art (The Met)**.
   - Afternoon: Explore **Central Park**, bike or walk.
   - Evening: Watch Broadway musical (book tickets in advance).

#### **Day 4: Modern NYC Experience**
   - Morning: Visit **Empire State Building** or **Top of the Rock** observation deck.
   - Afternoon: Shopping (Fifth Avenue, SOHO).
   - Evening: Try NYC specialties (pizza, cheesecake).

#### **Day 5: Brooklyn and Surroundings**
   - Morning: Visit Brooklyn, explore **Brooklyn Museum** or **Brooklyn Botanic Garden**.
   - Afternoon: Walk through **Williamsburg**, experience local artistic atmosphere.
   - Evening: Return to Manhattan, enjoy final night.

#### **Day 6: Departure or Extended Tour**
   - Arrange departure based on flight time, or continue to nearby cities (Washington DC, Boston).

---

### **IV. Budget Reference**
1. **Flights**: Round trip ~$700-1200 USD (varies by season and airline).
2. **Accommodation**: Mid-range hotel ~$200-350 USD per night.
3. **Dining**: Daily ~$40-70 USD.
4. **Attraction Tickets**:
   - Statue of Liberty ferry: ~$24.
   - The Met: Suggested donation $25.
   - Empire State Building: ~$44.
5. **Transportation**: MetroCard $34, airport transfers $20-$40.

---

### **V. Important Notes**
1. **Time Difference**: NYC is 13 hours behind Beijing (12 hours during daylight saving).
2. **Tipping Culture**: Restaurants, taxis require 15%-20% tips.
3. **Safety**: Keep personal belongings secure, avoid remote areas at night.

"""


'''
Command line usage:
python main.py --function rag_dispatcher

python main.py --function rag_dispatcher --rag_type self_rag
'''

def train():

    # Initialize model
    agent = TravelMind()  

    # # Launch UI  
    # launch_ui(agent)


    # agent.chat()



    # args = SFTArguments()  # Use parse_args to get arguments
    trainer = SFTTrainer(travel_agent = agent)

    processor = TravelQAProcessor(agent.tokenizer)

    processor.load_dataset_from_hf(DATA_PATH)

    trainer.max_length = processor.max_length
    print("trainer.max_length = ", trainer.max_length)



    processed_data = processor.prepare_training_features()

    print("mapping over")



    keys = list(processed_data.keys())

    print("keys = ", keys)

    # Adapt dataset size based on available data
    dataset_size = len(processed_data["train"])
    train_size = min(dataset_size, 3)  # Use 3 samples for training
    eval_size = min(dataset_size - train_size, 2)  # Use 2 samples for eval
    
    trainer.train(
        train_dataset=processed_data["train"].select(range(train_size)),
        eval_dataset=processed_data["train"].select(range(train_size, train_size + eval_size))
    )
    
def inference():
    # model = SFTTrainer.load_trained_model(SFT_MODEL_PATH)
    
    agent = TravelMind(MODEL_PATH)
    
    # agent.chat()  
    
    agent.stream_chat("I want to travel.")
    
    
def use_rag():
    agent = TravelMind(MODEL_PATH)
    rag = RAG(agent = agent)
    
    # results = rag.query_db("train tickets")
    # print(results)
    
    rag.rag_chat()
    
    
    # I want to go to Florida, I am now in the New York. Please help me book a hotel in Floria. I will arrive at 12:36:42 and leave at 22:43:12, my budget is 5000 dollars.


def use_city_rag():
   
   rag = CityRAG()
   rag.query("Help me plan a travel itinerary for Shanghai")
      
   # except Exception as e:
   #    print("Error building RAG object: ", str(e))
      
      
      
async def use_rag_dispatcher(rag_type:str="self_rag"):
    rag_dispatcher = RAGDispatcher(rag_type=rag_type)
   
   
    answer = await rag_dispatcher.dispatch("Help me plan a 3-day tour itinerary for Guangzhou")
    
    print("final answer  = ", answer)

def use_rag_web_demo():
    
    from src.ui.rag_web_demo import initialize_rag, create_demo
    rag_system = initialize_rag()  
    demo = create_demo(rag_system)  
    demo.launch(  
        server_name="0.0.0.0",  
        server_port=7860,  
        share=False,  
        favicon_path="./travel_icon.png"  
    )  
    
    
def use_agent():
    agent = MyAgent()
    

    
    result = agent.get_final_plan(PLAN_EXAMPLE)
    
    print(result)
    
    
async def parse_arguments(default_func = "use_agent"):
      parser = argparse.ArgumentParser(description="Travel Agent: Choose the function you wang to display ~")
      parser.add_argument(
         "--function", 
         type=str, 
         default = default_func, 
         help="Choose the function from [train, inference, use_rag, use_agent, use_rag_web_demo, rag_dispatcher]"
         )
      
      parser.add_argument(
         "--rag_type", 
         type=str, 
         default = "self_rag", 
         help="This is useful when --function==rag_dispatcher, Choose the RAG type from [self_rag, rag, mem_walker]"
         )

      # parser.add_argument(
      #    "--model", type=str, default="gpt-4o", help="model used for decoding. Please select from [gpt-4o, gpt-4o-mini]"
      # )
      

      
      args = parser.parse_args()
      
      if args.function == "train":
          train()

      elif args.function == "inference":
          inference()

      elif args.function == "use_rag":
          use_rag()

      elif args.function == "use_agent":
          use_agent()
      elif args.function == "rag_dispatcher":
          await use_rag_dispatcher(args.rag_type)
      elif args.function == "use_rag_web_demo":
          use_rag_web_demo()
          
          
    
if __name__ == "__main__":
    # inference()
    # use_rag()
    # use_rag_web_demo()
   #  use_agent()
   asyncio.run(parse_arguments())
   
   # use_city_rag()