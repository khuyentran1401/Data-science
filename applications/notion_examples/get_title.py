from notion.client import NotionClient

client = NotionClient(token_v2="02a65af9ae61ae50bc17f3333d251e2580d9fd1293bdc68990a474372a458a56b63b1b2868f47e5b21e947ddc093634ce8442de2390f14bcdc54bbaa0d6d2921fad82538d01c9d78c9931f6daea5")

page = client.get_block("https://www.notion.so/1063a61b0a4548959373f995b008d13e?v=6fba291866984949abf7c2c6e2e85ae1")
print("Page title is", page.title)

page.title = "My connections"
print("New page title is", page.title)

