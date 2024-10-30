from pinecone import Pinecone, ServerlessSpec
import os
import time


class PineconeManager:
    def __init__(self, api_key):
        self.pinecone_client = Pinecone(api_key=api_key,
                                        spec=ServerlessSpec(cloud=os.getenv('PINECONE_CLOUD', 'aws'),
                                        region=os.getenv('PINECONE_REGION', 'us-east-1')))
        self.indexes = {}
        self.spec = ServerlessSpec(cloud=os.getenv('PINECONE_CLOUD', 'aws'),
                                   region=os.getenv('PINECONE_REGION', 'us-east-1'))

    def create_index(self, index_name, dimension, spec=None):
        if spec is None:
            spec = self.spec
        if index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(name=index_name, dimension=dimension, metric="cosine", spec=spec)
            while not self.pinecone_client.describe_index(index_name).status['ready']:
                time.sleep(1)
        self.indexes[index_name] = self.pinecone_client.Index(index_name)
        return self.indexes[index_name]
    
    def get_index(self, index_name):
        if index_name in self.indexes:
            return self.indexes[index_name]
        else:
            raise ValueError(f"Index '{index_name}' does not exist.")



