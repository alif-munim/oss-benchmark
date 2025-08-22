from bs4 import BeautifulSoup
import requests
import csv

def scrape_eurorad_case(url):
    # Send an HTTP request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return
    
    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Initialize an empty dictionary to store the scraped data
    scraped_data = {}
    
    # List of sections to scrape
    sections_to_scrape = ["Section", "CLINICAL HISTORY", "IMAGING FINDINGS", "DISCUSSION", "DIFFERENTIAL DIAGNOSIS LIST", "FINAL DIAGNOSIS"]
    
    # Loop through each section and scrape its content
    for section in sections_to_scrape:
        # Find the HTML element containing the section
        section_element = soup.find("h2", string=section)  # Adjust the tag and attribute based on the actual HTML structure
        
        # Check if the section exists
        if section_element:
            # Find the content immediately following the section header
            content = section_element.find_next_sibling()  # Adjust this based on the actual HTML structure
            
            # Store the scraped content in the dictionary
            if content:
                scraped_data[section] = content.text.strip()
        else:
            print(f"Section '{section}' not found.")
    
    return scraped_data

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(["Section", "Content"])
        
        # Write the data
        for section, content in data.items():
            writer.writerow([section, content])

# URL of the Eurorad case
url = "https://www.eurorad.org/case/18283"

# Scrape the data
scraped_data = scrape_eurorad_case(url)

# Save the scraped data to a CSV file
save_to_csv(scraped_data, 'eurorad_case.csv')

print("Data has been saved to 'eurorad_case.csv'")