import requests  
from bs4 import BeautifulSoup  
import os  





# Define list of cities to crawl    
cities = ["Beijing", "Shanghai", "Guangzhou", "Chengdu"]    

# Create a directory to save files    
os.makedirs("tour_pages", exist_ok=True)  

# Basic Ctrip travel page URL format (needs adjustment based on actual pages)    
base_url = "https://you.ctrip.com/travels/"  

for city in cities:  
    # Convert city name to URL parameter (needs encoding)    
    city_encoded = requests.utils.quote(city)  
    url = f"{base_url}{city_encoded}"  

    print(f"Crawling: {url}")    
    response = requests.get(url)  

    if response.status_code == 200:  
        soup = BeautifulSoup(response.text, 'html.parser')  
        
        
    
        # Parse body content (adjust selector based on actual webpage structure)    
        content = soup.find('div', class_='content')  # Example selector    
        if content:  
            # Save to corresponding file    
            file_path = os.path.join("tour_pages", f"{city}.txt")  
            with open(file_path, 'w', encoding='utf-8') as f:  
                f.write(content.get_text(strip=True))  
                print(f"Saved travel information for {city} to {file_path}")    
        else:  
            print(f"Travel content for {city} not found")    
    else:  
        print(f"Request failed: {response.status_code} - {url}")  

print("Crawling completed")    