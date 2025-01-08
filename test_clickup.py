import requests
import json
import apis

def test_clickup_access():
    headers = {
        'Authorization': f'Bearer {apis.CLICKUP_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Get all spaces
    print("\nGetting all spaces...")
    spaces_response = requests.get(
        f'https://api.clickup.com/api/v2/team/{apis.CLICKUP_TEAM_ID}/space',
        headers=headers
    )
    print(f"Spaces response: {json.dumps(spaces_response.json(), indent=2)}")
    
    # Get all lists in each space
    print("\nGetting all lists in each space...")
    for space in spaces_response.json()['spaces']:
        print(f"\nSpace: {space['name']} (ID: {space['id']})")
        lists_response = requests.get(
            f'https://api.clickup.com/api/v2/space/{space["id"]}/list',
            headers=headers
        )
        print(f"Lists response: {json.dumps(lists_response.json(), indent=2)}")
    
    # Test task creation
    print("\nTesting task creation in o:produkt list...")
    task_data = {
        'name': 'Test Task - Please Delete',
        'description': 'This is a test task to verify bot configuration.'
    }
    
    create_response = requests.post(
        f'https://api.clickup.com/api/v2/list/{apis.CLICKUP_DEFAULT_LIST_ID}/task',
        headers=headers,
        json=task_data
    )
    print(f"Task creation response: {json.dumps(create_response.json(), indent=2)}")

if __name__ == "__main__":
    test_clickup_access() 