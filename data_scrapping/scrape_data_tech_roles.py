import csv
from jobspy import scrape_jobs
import pandas as pd

TECH_ROLES = [
    "Software Engineer", "Frontend Engineer", "Backend Engineer",
    "Full Stack Engineer", "Machine Learning Engineer", "Data Engineer",
    "AI Engineer", "Cloud Engineer", "DevOps Engineer", "Platform Engineer",
    "Embedded Engineer", "Firmware Engineer", "Hardware Engineer",
    "Systems Engineer", "Security Engineer", "Site Reliability Engineer",
    "Mobile Engineer", "iOS Engineer", "Android Engineer",
    "QA Engineer", "Test Automation Engineer"
]

dfs = []
role_counts = {}

for role in TECH_ROLES:
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
    "all_tech_jobs.csv",
    quoting=csv.QUOTE_NONNUMERIC,
    escapechar="\\",
    index=False
)

print("\n✅ Saved to: all_tech_jobs.csv")

