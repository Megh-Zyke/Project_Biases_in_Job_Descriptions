from serpapi import GoogleSearch
import csv
import os

def fetch_jobs(query: str, location: str, api_key: str, num_results: int = 10):
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "api_key": api_key,
        "google_domain": "google.com",
        "hl": "en",
        "gl": "us"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    jobs = results.get("jobs_results", [])
    
    extracted = []
    for job in jobs[:num_results]:
        extracted.append({
            "title": job.get("title"),
            "company": job.get("company_name"),
            "location": job.get("location"),
            "description": job.get("description"),
            "link": job.get("share_link")
        })
    return extracted


queries = [
    # 💬 Customer-Facing Roles
    "Sales / Business Development",
    "Marketing / Brand Management",
    "Retail / Store Management",
    "Public Relations / Communications",
    "Event Planning / Coordination",
    "Customer Success / Client Relations",
    "Front Desk / Reception / Office Administration",
    "Travel & Tourism",
    "Real Estate / Property Management",
    "Healthcare / Nursing / Patient Care",
    "Video Production / Editing",
    "Photography / Videography",
    "Animation / Motion Graphics",
    "Social Media Management",
    "Advertising / Campaign Strategy",
    "Fashion / Styling / Merchandising",
    "Music / Sound Design",
    "Art / Illustration",
    "Interior Design / Architecture",
    "Maintenance / Facilities Management",
    "Construction / Skilled Trades",
    "Logistics / Delivery / Transportation",
    "Food & Beverage / Culinary",
    "Beauty / Wellness / Spa Services",
    "Fitness / Personal Training / Sports Coaching",
    "Security / Safety Services",
    "Cleaning / Housekeeping",
    "Childcare / Elderly Care",
    "Landscaping / Gardening"
]
# Example usage
locations = [
    "India", "United Kingdom", "Canada", "Remote", "Germany", "United States",
    "Australia", "France", "Italy", "Spain", "Netherlands", "Sweden",
    "Switzerland", "Brazil", "Mexico", "Japan", "Singapore", "United Arab Emirates"
]

api_key = "9b0bbfc749628df513ab812e123fdbc4d4a3669f8219a07da93d207ae931025d"

num_results_per_location = 200 // len(locations) - 2
total_jobs = []

for query in queries:
 for loc in locations:
    print(f"Fetching jobs in {loc}...")
    jobs = fetch_jobs( query, loc, api_key, num_results=3 )
    total_jobs.extend(jobs)

print(f"Total jobs fetched this run: {len(total_jobs)}")

# === Save results (append mode) ===
out_path = os.path.join(os.path.dirname(__file__), "jobs_export.csv")
file_exists = os.path.exists(out_path)



fieldnames = ["title", "company", "location", "description", "link"]

with open(out_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    # Write header only if file is new
    if not file_exists:
        writer.writeheader()
    
    for job in total_jobs:
        writer.writerow({k: (job.get(k) or "") for k in fieldnames})

print(f"Appended {len(total_jobs)} new jobs to {out_path}")
