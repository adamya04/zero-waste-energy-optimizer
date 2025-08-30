import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.web import WebSiteManagementClient
from azure.mgmt.sql import SqlManagementClient

# Azure configuration
subscription_id = "b8858451-615c-4e82-8ed6-44a0aff019aa"  # Replace
resource_group = "TronWasteResourceGroup"
app_service_plan = "TronWastePlan"
web_app_name = "tronwasteoptimizer"
sql_server_name = "tronwastesqlserver"
sql_database_name = "tronwastedb"
location = "East US"

credential = DefaultAzureCredential()
resource_client = ResourceManagementClient(credential, subscription_id)
web_client = WebSiteManagementClient(credential, subscription_id)
sql_client = SqlManagementClient(credential, subscription_id)

# Create resource group
resource_client.resource_groups.create_or_update(resource_group, {"location": location})

# Create App Service Plan
plan = web_client.app_service_plans.begin_create_or_update(
    resource_group, app_service_plan, {"location": location, "sku": {"name": "B1", "tier": "Basic"}}
).result()

# Create Web App
web_client.web_apps.begin_create_or_update(
    resource_group, web_app_name, {
        "location": location,
        "server_farm_id": plan.id,
        "https_only": True
    }
).result()

# Create SQL Server and Database
sql_client.servers.begin_create_or_update(
    resource_group, sql_server_name, {"location": location, "administrator_login": "adminuser", "administrator_login_password": "your_password"}  # Replace
).result()
sql_client.databases.begin_create_or_update(
    resource_group, sql_server_name, sql_database_name, {"location": location}
).result()

# Deploy backend code
with open("backend/requirements.txt", "rb") as f:
    web_client.web_apps.begin_update_application_settings(
        resource_group, web_app_name, {"WEBSITE_RUN_FROM_PACKAGE": "1", "PYTHON_VERSION": "3.11"}
    ).result()
os.system(f"az webapp deploy --resource-group {resource_group} --name {web_app_name} --src-path backend")

print(f"Deployed to Azure at http://{web_app_name}.azurewebsites.net")