import requests

def get_github_skills(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)

    if response.status_code != 200:
        return []

    repos = response.json()
    languages = set()

    for repo in repos:
        if repo["language"]:
            languages.add(repo["language"].lower())

    return list(languages)