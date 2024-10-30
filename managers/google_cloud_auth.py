from google.cloud import aiplatform
from google.oauth2 import service_account

class GoogleCloudAuth:
    def __init__(self, project_id, region, key_path):
        self.project_id = project_id
        self.region = region
        self.credentials = service_account.Credentials.from_service_account_file(key_path)

    def authenticate(self):
        aiplatform.init(project=self.project_id, credentials=self.credentials, location=self.region)
        print("Authenticated with Google Cloud Platform")
