import csv
from jobspy import scrape_jobs
import pandas as pd



BUSINESS_ROLES = [
    
    "Sales",
    "Account Executive",
    "Account Manager",
    "Business Development",
    "Customer Success",

    # Marketing
    "Marketing",
    "Digital Marketing",
    "Product Marketing",
    "Content Marketing",
    "Social Media",

    # Finance / Accounting
    "Financial Analyst",
    "Accountant",
    "Finance Manager",
    "Controller",
    "FP&A",

    # Human Resources
    "Human Resources",
    "HR Manager",
    "Recruiter",
    "Talent Acquisition",
    "People Operations",

    # Operations / Supply Chain
    "Operations",
    "Operations Manager",
    "Supply Chain",
    "Procurement",
    "Logistics",

    # Project / Product Management
    "Project Manager",
    "Program Manager",
    "Product Manager",
    "Scrum Master",
]


print(len(BUSINESS_ROLES))
dfs = []
role_counts = {}

for role in BUSINESS_ROLES:
    print(f"Scraping role: {role}")

    try:
        df = scrape_jobs(
            site_name=[
                "indeed", "linkedin", "google",
                "zip_recruiter", "glassdoor"],
            search_term=role,
            results_wanted=80,
            hours_old=168,  
            country_indeed="USA",
        )

        if df is not None:
            dfs.append(df)
            role_counts[role] = len(df)
        else:
            role_counts[role] = 0

    except Exception as e:
        print(f"Error scraping {role}: {e}")
        role_counts[role] = 0


all_jobs = pd.concat(dfs, ignore_index=True).drop_duplicates()
print(f"\n✅ TOTAL jobs scraped: {len(all_jobs)}\n")

for role, count in role_counts.items():
    print(f"{role}: {count}")


all_jobs.to_csv(
    "all_non_tech_jobs.csv",
    quoting=csv.QUOTE_NONNUMERIC,
    escapechar="\\",
    index=False
)

print("\nSaved to: all_non_tech_jobs.csv")

