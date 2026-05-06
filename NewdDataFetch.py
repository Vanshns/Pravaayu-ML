# import requests
# import json
# import os

# # --- Configuration ---
# BASE_LIST_URL = "https://www.pravaayu.com/api/crf-pat-file-list"
# BASE_DETAIL_URL = "https://www.pravaayu.com/api/crf/"
# JSON_FILE = "patients.json"

# # Parameters for the list API
# params = {
#     "startDate": "2026-05-01",
#     "endDate": "2026-05-05",
#     "limit": 50
# }

# def load_existing_data():
#     if os.path.exists(JSON_FILE):
#         with open(JSON_FILE, "r") as f:
#             try:
#                 return json.load(f)
#             except json.JSONDecodeError:
#                 return []
#     return []

# def save_data(data_list):
#     with open(JSON_FILE, "w") as f:
#         json.dump(data_list, f, indent=4)

# def fetch_patient_details(patient_id, file_no):
#     # Logic: prioritize file_no if it's not "-"
#     if file_no and file_no != "-":
#         url = f"{BASE_DETAIL_URL}{file_no}"
#     else:
#         url = f"{BASE_DETAIL_URL}{patient_id}?patientId={patient_id}"
    
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             return response.json()
#     except Exception as e:
#         print(f"Error fetching details for {patient_id}: {e}")
#     return None

# def main():
#     all_patients = load_existing_data()
#     current_page = 1
    
#     while True:
#         print(f"Fetching Page {current_page}...")
#         params["page"] = current_page
        
#         response = requests.get(BASE_LIST_URL, params=params)
#         if response.status_code != 200:
#             print(f"Failed to fetch list. Status: {response.status_code}")
#             break
            
#         result = response.json()
#         data_list = result.get("data", [])
        
#         # Stop if data is empty
#         if not data_list:
#             print("No more data found. Finishing...")
#             break
            
#         for item in data_list:
#             p_id = item.get("patientID")
#             f_no = item.get("file_no")
            
#             print(f"  --> Getting details for: {f_no if f_no != '-' else p_id}")
#             details = fetch_patient_details(p_id, f_no)
            
#             if details:
#                 all_patients.append(details)
        
#         # Save after every page to prevent data loss if it crashes
#         save_data(all_patients)
#         current_page += 1

# if __name__ == "__main__":
#     main()

import requests
import json
import os

# --- Configuration ---
BASE_LIST_URL = "https://www.pravaayu.com/api/crf-pat-file-list"
BASE_DETAIL_URL = "https://www.pravaayu.com/api/crf/"
JSON_FILE = "patients.json"

params = {
    "startDate": "2026-05-01",
    "endDate": "2026-05-05",
    "limit": 50
}

def load_existing_data():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_data(data_list):
    with open(JSON_FILE, "w") as f:
        json.dump(data_list, f, indent=4)

def fetch_patient_details(patient_id, file_no):
    if file_no and file_no != "-":
        url = f"{BASE_DETAIL_URL}{file_no}"
    else:
        url = f"{BASE_DETAIL_URL}{patient_id}?patientId={patient_id}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching details for {patient_id}: {e}")
    return None

def main():
    all_patients = load_existing_data()
    
    # Create a set of existing IDs for O(1) lookup speed
    # We check for 'patientID' or 'id' depending on your JSON structure
    existing_ids = {p.get('patientID') or p.get('id') for p in all_patients if p}
    
    current_page = 1
    new_records_added = 0
    
    while True:
        print(f"Fetching Page {current_page}...")
        params["page"] = current_page
        
        response = requests.get(BASE_LIST_URL, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch list. Status: {response.status_code}")
            break
            
        result = response.json()
        data_list = result.get("data", [])
        
        if not data_list:
            print("No more data found. Finishing...")
            break
            
        for item in data_list:
            p_id = item.get("patientID")
            f_no = item.get("file_no")
            
            # --- DUPLICATE CHECK ---
            if p_id in existing_ids:
                print(f"  SKIPPING: {p_id} (Already in file)")
                continue
            
            print(f"  --> Getting details for: {f_no if f_no != '-' else p_id}")
            details = fetch_patient_details(p_id, f_no)
            
            if details:
                all_patients.append(details)
                # Add to set so we don't add it again if it appears twice in the API
                existing_ids.add(p_id) 
                new_records_added += 1
        
        save_data(all_patients)
        current_page += 1

    print(f"Task Complete. Added {new_records_added} new unique records.")

if __name__ == "__main__":
    main()